import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv
import argparse
import tempfile
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torchvision import transforms

_cache_dir = os.path.join(tempfile.gettempdir(), "stable_ginv_cache")
_mpl_config_dir = os.path.join(_cache_dir, "matplotlib")
_xdg_cache_dir = os.path.join(_cache_dir, "xdg")
os.makedirs(_mpl_config_dir, exist_ok=True)
os.makedirs(_xdg_cache_dir, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _mpl_config_dir)
os.environ.setdefault("XDG_CACHE_HOME", _xdg_cache_dir)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import functions.consts as consts
from functions.Dataset import load_dataset
from functions.io_utils import parse_prefixes_with_fracs
from helper.Network import get_model, weights_init
from functions.masking import build_gradient_mask
from helper.metrics import compute_jacobian_rank_sweep


_HPC_ROOT = "/work3/s234843/bachelor"
_TORCH_DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
}


def _storage_root():
    return _HPC_ROOT if os.access(_HPC_ROOT, os.R_OK | os.W_OK | os.X_OK) else None


def _data_path():
    root = _storage_root()
    return os.path.join(root, "datasets") if root else "./data"


def _save_dir():
    root = _storage_root()
    return os.path.join(root, "results") if root else "./results"


def _split_list(lst, n_chunks):
    """Split lst into n_chunks as evenly as possible."""
    chunks = [[] for _ in range(n_chunks)]
    for i, x in enumerate(lst):
        chunks[i % n_chunks].append(x)
    return chunks


def _torch_dtype(dtype_name):
    try:
        return _TORCH_DTYPES[dtype_name]
    except KeyError:
        raise ValueError(f"Unsupported dtype: {dtype_name}") from None


def _seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(requested_device, worker_rank=None):
    if requested_device == "cuda":
        if torch.cuda.is_available():
            if worker_rank is None:
                return "cuda:0"
            return f"cuda:{worker_rank % torch.cuda.device_count()}"
        return "cpu"

    if requested_device.startswith("cuda"):
        if torch.cuda.is_available():
            if worker_rank is None:
                return requested_device
            return f"cuda:{worker_rank % torch.cuda.device_count()}"
        print(f"CUDA requested via --device {requested_device}, but CUDA is unavailable; using CPU.")
        return "cpu"

    return requested_device


def _new_rank_results(row_counts, qr_pivot):
    return (
        {rows: [] for rows in row_counts},
        {rows: [] for rows in row_counts} if qr_pivot else None,
    )


def _summarize_rank_results(row_counts, rank_results, rank_results_qr):
    xs = list(row_counts)
    mean_ranks = [float(np.mean(rank_results[rows])) for rows in xs]
    std_ranks = [float(np.std(rank_results[rows])) for rows in xs]

    mean_ranks_qr, std_ranks_qr = None, None
    if rank_results_qr is not None:
        mean_ranks_qr = [float(np.mean(rank_results_qr[rows])) for rows in xs]
        std_ranks_qr = [float(np.std(rank_results_qr[rows])) for rows in xs]

    return {
        "rank_results": rank_results,
        "rank_results_qr": rank_results_qr,
        "xs": xs,
        "mean_ranks": mean_ranks,
        "std_ranks": std_ranks,
        "mean_ranks_qr": mean_ranks_qr,
        "std_ranks_qr": std_ranks_qr,
    }


def _print_rank_summary(dtype_name, select_mode, run):
    print(f"\nPer-sample ranks ({select_mode}, dtype={dtype_name}):")
    for rows, mr, sr in zip(run["xs"], run["mean_ranks"], run["std_ranks"]):
        ranks = run["rank_results"][rows]
        print(f"  rows={rows:6d}: {ranks}  mean={mr:.2f}  std={sr:.2f}")

    if run["rank_results_qr"] is not None:
        print(f"\nPer-sample ranks (qr_pivot, dtype={dtype_name}):")
        for rows, mr, sr in zip(run["xs"], run["mean_ranks_qr"], run["std_ranks_qr"]):
            ranks = run["rank_results_qr"][rows]
            print(f"  rows={rows:6d}: {ranks}  mean={mr:.2f}  std={sr:.2f}")


def _print_mask_debug(args, net, original_dy_dx, keep_ids, entry_masks):
    named_params = list(net.named_parameters())
    print("\n=== MASK DEBUG ===", flush=True)
    print("mask_mode:", args.mask_mode, flush=True)
    print("entry_masks is None:", entry_masks is None, flush=True)
    print("num keep_ids:", 0 if keep_ids is None else len(keep_ids), flush=True)
    if keep_ids is not None:
        keep_ids_set = set(keep_ids)
        total_entries = sum(g.numel() for g in original_dy_dx if g is not None)
        observed_entries = sum(
            g.numel() for i, g in enumerate(original_dy_dx)
            if g is not None and i in keep_ids_set
        )
        print(f"observed_entries={observed_entries}, total_entries={total_entries}, "
              f"kept_fraction={observed_entries/total_entries:.6f}", flush=True)
        kept_names   = [name for i, (name, _) in enumerate(named_params) if i in keep_ids_set]
        skipped_names = [name for i, (name, _) in enumerate(named_params) if i not in keep_ids_set]
        print(f"kept tensors ({len(kept_names)}): {kept_names}", flush=True)
        if skipped_names:
            print(f"skipped tensors ({len(skipped_names)}): {skipped_names}", flush=True)
    print("=== END MASK DEBUG ===\n", flush=True)


def _worker_core(args, sample_indices, device, row_counts, prefixes, prefix_layer_fracs,
                 progress_fn=None, debug=False):
    """
    Compute Jacobian rank for each sample in sample_indices on device.
    Returns (results, results_qr) where results_qr is None when args.qr_pivot is False.
    progress_fn(1) is called after each AD pass (J build) and after each rank computation.
    debug: if True, prints mask debug info for the first sample.
    """
    dst, channel, num_classes, shape_img = load_dataset(args.dataset, _data_path())
    dtype = _torch_dtype(args.dtype)

    net = get_model(args.network, channel=channel, num_classes=num_classes,
                    input_size=shape_img, pretrained=args.pretrained)
    if not args.pretrained and args.network in ("LeNet", "LeNet_bigger", "MediumCNN", "BiggerCNN"):
        net.apply(weights_init)
    net = net.to(device=device, dtype=dtype)
    net.eval()

    tt = transforms.Compose([transforms.ToTensor()])
    criterion = nn.CrossEntropyLoss().to(device)

    if args.pretrained and channel == 3:
        dm = torch.tensor(consts.imagenet_mean, device=device, dtype=dtype).view(1, channel, 1, 1)
        ds = torch.tensor(consts.imagenet_std, device=device, dtype=dtype).view(1, channel, 1, 1)
    else:
        dm = torch.tensor(getattr(consts, f'{args.dataset.lower()}_mean'), device=device, dtype=dtype).view(1, channel, 1, 1)
        ds = torch.tensor(getattr(consts, f'{args.dataset.lower()}_std'), device=device, dtype=dtype).view(1, channel, 1, 1)

    results, results_qr = _new_rank_results(row_counts, args.qr_pivot)
    n_samples = len(sample_indices)

    for local_i, idx in enumerate(sample_indices):
        gt_data = tt(dst[idx][0]).to(device=device, dtype=dtype).unsqueeze(0)
        gt_label = torch.tensor([dst[idx][1]], dtype=torch.long, device=device)

        gt_data_norm = (gt_data - dm) / ds
        out = net(gt_data_norm)
        loss = criterion(out, gt_label)
        dy_dx = torch.autograd.grad(loss, net.parameters())
        original_dy_dx = [g.detach() for g in dy_dx]

        grad_norm = sum(g.norm().item() for g in original_dy_dx if g is not None)
        has_nan = any(torch.isnan(g).any().item() for g in original_dy_dx if g is not None)
        has_inf = any(torch.isinf(g).any().item() for g in original_dy_dx if g is not None)
        tqdm.write(
            f"[{local_i + 1}/{n_samples}] idx={idx}  grad_norm={grad_norm:.4f}"
            f"  nan={has_nan}  inf={has_inf}  — building J...",
            file=sys.stderr,
        )

        keep_ids, entry_masks = build_gradient_mask(
            method=args.method,
            mask_mode=args.mask_mode,
            net=net,
            original_dy_dx=original_dy_dx,
            prefixes=prefixes,
            prefix_layer_fracs=prefix_layer_fracs,
            gradsize_topk=args.gradsize_topk,
            gradsize_topfrac=args.gradsize_topfrac,
            gradsize_metric="l2",
        )

        if debug and local_i == 0:
            _print_mask_debug(args, net, original_dy_dx, keep_ids, entry_masks)

        sweep, sweep_qr = compute_jacobian_rank_sweep(
            net=net,
            x_norm=gt_data_norm,
            y=gt_label,
            criterion=criterion,
            keep_ids=keep_ids,
            entry_masks=entry_masks,
            row_counts=row_counts,
            select_mode=args.jacobian_select_mode,
            qr_pivot=args.qr_pivot,
            device_for_J=device,
            print_svd_info=args.print_svd_info,
            j_progress_fn=progress_fn,
            rank_progress_fn=progress_fn,
            independent=args.independent,
            normalize_rows=not args.no_normalisation,
        )
        for rows in row_counts:
            jac_rank, jac_shape, _, _ = sweep[rows]
            results[rows].append(jac_rank)
            if results_qr is not None:
                results_qr[rows].append(sweep_qr[rows][0])
            qr_str = f"  rank_qr={sweep_qr[rows][0]}" if sweep_qr is not None else ""
            tqdm.write(
                f"[{local_i + 1}/{n_samples}] rows={rows}: rank={jac_rank}"
                f"  shape={jac_shape}{qr_str}",
                file=sys.stderr,
            )

        del gt_data, gt_label, gt_data_norm, out, loss, dy_dx, original_dy_dx
        if torch.cuda.is_available() and device.startswith("cuda"):
            torch.cuda.empty_cache()

    return results, results_qr


def _mp_worker(rank, world_size, args, sample_chunks, row_counts, prefixes, prefix_layer_fracs,
               shared_results, progress_counter, progress_lock):
    """mp.spawn target: runs _worker_core on one GPU."""
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())

    device = _resolve_device(args.device, worker_rank=rank)
    _seed_all(args.run_id + 1 + rank)

    def progress_fn(n):
        with progress_lock:
            progress_counter.value += n

    results, results_qr = _worker_core(
        args, sample_chunks[rank], device, row_counts, prefixes, prefix_layer_fracs,
        progress_fn=progress_fn,
        debug=(rank == 0),
    )
    shared_results[rank] = (results, results_qr)


def _run_serial(args, sample_indices, row_counts, prefixes, prefix_layer_fracs, total_steps, dtype_name):
    device = _resolve_device(args.device)
    with tqdm(total=total_steps, desc=f"Jacobian rank ({dtype_name})",
              unit="step", dynamic_ncols=True) as pbar:
        return _worker_core(
            args, sample_indices, device, row_counts, prefixes, prefix_layer_fracs,
            progress_fn=lambda n: pbar.update(n),
            debug=True,
        )


def _run_parallel(args, sample_indices, row_counts, prefixes, prefix_layer_fracs, total_steps, dtype_name):
    world_size = min(args.num_workers, len(sample_indices))
    sample_chunks = _split_list(sample_indices, world_size)

    manager = mp.Manager()
    shared_results = manager.dict()
    progress_counter = manager.Value("i", 0)
    progress_lock = manager.Lock()

    spawn_ctx = mp.spawn(
        _mp_worker,
        args=(world_size, args, sample_chunks, row_counts, prefixes, prefix_layer_fracs,
              shared_results, progress_counter, progress_lock),
        nprocs=world_size,
        join=False,
    )

    last = 0
    with tqdm(total=total_steps, desc=f"Jacobian rank ({dtype_name})") as pbar:
        while not spawn_ctx.join(timeout=0.2):
            current = progress_counter.value
            if current > last:
                pbar.update(current - last)
                last = current
        current = progress_counter.value
        if current > last:
            pbar.update(current - last)

    rank_results, rank_results_qr = _new_rank_results(row_counts, args.qr_pivot)
    for worker_id in range(world_size):
        worker_results, worker_results_qr = shared_results[worker_id]
        for rows in row_counts:
            rank_results[rows].extend(worker_results[rows])
            if rank_results_qr is not None:
                rank_results_qr[rows].extend(worker_results_qr[rows])

    return rank_results, rank_results_qr


def _run_dtype(args, dtype_name, sample_indices, row_counts, prefixes, prefix_layer_fracs, total_steps):
    args.dtype = dtype_name
    print(f"\n=== Running dtype: {dtype_name} ===")
    _seed_all(args.run_id + 1)

    if args.num_workers == 1:
        rank_results, rank_results_qr = _run_serial(
            args, sample_indices, row_counts, prefixes, prefix_layer_fracs, total_steps, dtype_name
        )
    else:
        rank_results, rank_results_qr = _run_parallel(
            args, sample_indices, row_counts, prefixes, prefix_layer_fracs, total_steps, dtype_name
        )

    run = _summarize_rank_results(row_counts, rank_results, rank_results_qr)
    _print_rank_summary(dtype_name, args.jacobian_select_mode, run)
    return run


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--network", type=str, default="resnet18")
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--method", type=str, default="idlg", choices=["idlg", "masked"])
    parser.add_argument("--pretrained", action="store_true",
                        help="Load ImageNet-pretrained weights (affects normalization).")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"],
                        help="Floating-point dtype used for the network, input, gradients, "
                             "and Jacobian construction. Default preserves previous behavior.")
    parser.add_argument("--both_dtypes", action="store_true",
                        help="Run the same sweep twice, once with float32 and once with float64, "
                             "and plot both rank curves in one graph.")

    parser.add_argument(
        "--mask_mode",
        type=str,
        default="gradsize_topfrac",
        choices=[
            "none",
            "all",
            "gradsize_topk",
            "gradsize_topfrac",
            "gradsize_topk_entries",
            "gradsize_topfrac_entries",
            "gradsize_topk_entries_layer",
            "gradsize_topfrac_entries_layer",
            "prefix",
            "prefix_topk",
            "prefix_topfrac",
            "prefix_topk_entries",
            "prefix_topfrac_entries",
            "prefix_topk_entries_layer",
            "prefix_topfrac_entries_layer",
        ],
    )
    parser.add_argument("--prefixes", type=str, default="conv1:1.0,layer1:1.0,layer2:1.0,layer3:1.0,fc:1.0")
    parser.add_argument("--gradsize_topk", type=int, default=20)
    parser.add_argument("--gradsize_topfrac", type=float, default=0.5)

    parser.add_argument("--row_counts", type=str,
                        default="3072,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000")
    parser.add_argument("--stepsize", type=int, default=None,
                        help="Step between row counts. Generates range from --min_row_count "
                             "(default: unknowns) to --max_row_count in increments of stepsize. "
                             "Overrides --row_counts when set.")
    parser.add_argument("--max_row_count", type=int, default=None,
                        help="Upper bound for row counts when --stepsize is used.")
    parser.add_argument("--min_row_count", type=int, default=None,
                        help="Smallest row count to sweep when --stepsize is used. "
                             "Defaults to unknowns (= C×H×W).")
    parser.add_argument("--jacobian_select_mode", type=str, default="topk_abs",
                        choices=["topk_abs", "first", "random", "layer_spread"])
    parser.add_argument("--qr_pivot", action="store_true",
                        help="After building J with --jacobian_select_mode, reorder its rows "
                             "via QR column pivoting so each slice J[:k] contains the k most "
                             "linearly independent rows from the pool.")
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--sample_indices", type=str, default=None,
                        help="Comma-separated dataset indices to use instead of random sampling, "
                             "e.g. --sample_indices 39508,23784. Overrides --num_samples and --run_id.")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of parallel workers (1 = serial, 2-4 = one process per GPU).")
    parser.add_argument("--print_svd_info", action="store_true",
                        help="Print sigma_max, atol, and the 10 smallest singular values "
                             "of the Jacobian at each row count.")
    parser.add_argument("--no_normalisation", action="store_true",
                        help="Skip row normalisation before computing matrix_rank. "
                             "The default relative threshold (max(M,N)*eps*sigma_max) "
                             "then reflects actual gradient magnitudes rather than directions.")
    parser.add_argument("--independent", action="store_true",
                        help="Build J independently for each row count (max_entries=k per k). "
                             "Theoretically correct: rank at k reflects exactly k gradient entries "
                             "selected by --jacobian_select_mode. Costs len(row_counts)x more "
                             "forward passes than the default pool-slice approach.")

    args = parser.parse_args()

    if not 1 <= args.num_workers <= 4:
        parser.error("--num_workers must be between 1 and 4")
    if not 0 < args.gradsize_topfrac <= 1:
        parser.error("--gradsize_topfrac must be in (0, 1]")
    if args.num_samples < 1:
        parser.error("--num_samples must be at least 1")

    if args.stepsize is not None:
        if args.max_row_count is None:
            parser.error("--max_row_count is required when --stepsize is used")
        if args.stepsize <= 0:
            parser.error("--stepsize must be a positive integer")
        if args.max_row_count <= 0:
            parser.error("--max_row_count must be a positive integer")
        if args.min_row_count is not None and args.min_row_count <= 0:
            parser.error("--min_row_count must be a positive integer")
        row_counts = None  # built after dataset load
    else:
        row_counts = [int(x.strip()) for x in args.row_counts.split(",") if x.strip()]
        if len(row_counts) == 0:
            parser.error("--row_counts must contain at least one positive integer")
        if any(rows <= 0 for rows in row_counts):
            parser.error("--row_counts values must all be positive")
    prefixes, prefix_layer_fracs = parse_prefixes_with_fracs(args.prefixes)
    if any(not 0 < frac <= 1 for frac in prefix_layer_fracs.values()):
        parser.error("prefix fractions in --prefixes must be in (0, 1]")

    save_dir = _save_dir()
    os.makedirs(save_dir, exist_ok=True)

    dst, channel, num_classes, shape_img = load_dataset(args.dataset, _data_path())

    if args.sample_indices is not None:
        sample_indices = [int(x.strip()) for x in args.sample_indices.split(",") if x.strip()]
        if any(i < 0 or i >= len(dst) for i in sample_indices):
            parser.error(f"--sample_indices contains an index out of range for dataset of size {len(dst)}")
    else:
        seed = args.run_id + 1
        _seed_all(seed)
        idx_shuffle = np.random.permutation(len(dst))
        sample_indices = idx_shuffle[:args.num_samples].tolist()

    unknowns = channel * shape_img[0] * shape_img[1]

    if args.stepsize is not None:
        min_row_count = args.min_row_count if args.min_row_count is not None else (unknowns // 1000) * 1000
        if min_row_count > args.max_row_count:
            parser.error(f"--min_row_count ({min_row_count}) exceeds --max_row_count ({args.max_row_count})")
        row_counts = list(range(min_row_count, args.max_row_count + 1, args.stepsize))
        if not row_counts or row_counts[-1] < args.max_row_count:
            row_counts.append(args.max_row_count)

    if unknowns not in row_counts:
        row_counts = sorted(row_counts + [unknowns])

    print(f"Unknowns: {unknowns}")
    print(f"Samples: {sample_indices}")
    dtype_values = ["float32", "float64"] if args.both_dtypes else [args.dtype]
    print(f"Dtype(s): {dtype_values}")

    # AD passes per sample: fwAD when unknowns < k (unknowns passes), else backward (k passes).
    rank_steps = len(row_counts) * (2 if args.qr_pivot else 1)
    if args.independent:
        j_steps = sum(min(k, unknowns) for k in row_counts)
    else:
        j_steps = unknowns  # one build at max(row_counts), always fwAD when max > unknowns
    total_steps = len(sample_indices) * (j_steps + rank_steps)

    runs = {}
    for dtype_name in dtype_values:
        runs[dtype_name] = _run_dtype(
            args, dtype_name, sample_indices, row_counts, prefixes, prefix_layer_fracs, total_steps
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dtype_label = "both_dtypes" if args.both_dtypes else dtype_values[0]
    base = (
        f"jac_rank_{args.network}_{args.dataset}_{args.method}_"
        f"{args.jacobian_select_mode}_{dtype_label}_ns{args.num_samples}_{timestamp}"
    )

    csv_path = os.path.join(save_dir, base + ".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["rows_used", "mean_rank", "std_rank",
                  "unknowns", "num_samples", "jacobian_select_mode", "dtype", "independent",
                  "sample_indices", "per_sample_ranks"]
        if args.qr_pivot:
            header += ["mean_rank_qr", "std_rank_qr", "per_sample_ranks_qr"]
        writer.writerow(header)
        sample_indices_str = ";".join(str(i) for i in sample_indices)
        for dtype_name, run in runs.items():
            rank_results = run["rank_results"]
            rank_results_qr = run["rank_results_qr"]
            for i, (x, m, s) in enumerate(zip(run["xs"], run["mean_ranks"], run["std_ranks"])):
                ranks_str = ";".join(str(r) for r in rank_results[x])
                row = [x, m, s, unknowns, args.num_samples, args.jacobian_select_mode, dtype_name, args.independent,
                       sample_indices_str, ranks_str]
                if rank_results_qr is not None:
                    ranks_qr_str = ";".join(str(r) for r in rank_results_qr[x])
                    row += [run["mean_ranks_qr"][i], run["std_ranks_qr"][i], ranks_qr_str]
                writer.writerow(row)

    plt.figure(figsize=(7, 5))
    for dtype_name, run in runs.items():
        legend_label = f"{dtype_name} (select={args.jacobian_select_mode}, samples={args.num_samples}"
        if args.method == "masked":
            legend_label += f", mask={args.mask_mode}"
        legend_label += ")"
        container = plt.errorbar(run["xs"], run["mean_ranks"], yerr=run["std_ranks"],
                                 marker="o", capsize=4, label=legend_label)
        if run["rank_results_qr"] is not None:
            color = container.lines[0].get_color()
            qr_label = f"{dtype_name} qr_pivot ({args.jacobian_select_mode} pool)"
            plt.errorbar(run["xs"], run["mean_ranks_qr"], yerr=run["std_ranks_qr"],
                         marker="s", capsize=4, linestyle="--", color=color, label=qr_label)
    plt.axhline(unknowns, linestyle=":", label=f"unknowns = {unknowns}")
    plt.xlabel("Number of Jacobian rows / gradients used")
    plt.ylabel("Average Jacobian rank")
    plt.title(f"Jacobian rank sweep: {args.network}, {args.dataset}, {args.method}")
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(save_dir, base + ".png")
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"\nSaved CSV to: {csv_path}")
    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
