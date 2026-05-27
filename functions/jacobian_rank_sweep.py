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



def _data_path():
    if os.access('/work3/s234843/bachelor', os.R_OK | os.W_OK | os.X_OK):
        return '/work3/s234843/bachelor/datasets'
    return './data'


def _split_list(lst, n_chunks):
    """Split lst into n_chunks as evenly as possible."""
    chunks = [[] for _ in range(n_chunks)]
    for i, x in enumerate(lst):
        chunks[i % n_chunks].append(x)
    return chunks


def _print_mask_debug(args, net, prefixes, prefix_layer_fracs, original_dy_dx, keep_ids, entry_masks):
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
    Returns {rows: [rank_list]}.
    progress_fn(1) is called after each (sample, row_count) step if provided.
    debug: if True, prints mask debug info for the first sample.
    """
    dst, channel, num_classes, shape_img = load_dataset(args.dataset, _data_path())

    net = get_model(args.network, channel=channel, num_classes=num_classes,
                    input_size=shape_img, pretrained=args.pretrained)
    if not args.pretrained and args.network in ("LeNet", "LeNet_bigger", "MediumCNN", "BiggerCNN"):
        net.apply(weights_init)
    net = net.to(device).double()
    net.eval()

    tt = transforms.Compose([transforms.ToTensor()])
    criterion = nn.CrossEntropyLoss().to(device)

    if args.pretrained and channel == 3:
        dm = torch.tensor(consts.imagenet_mean, device=device, dtype=torch.float64).view(1, channel, 1, 1)
        ds = torch.tensor(consts.imagenet_std, device=device, dtype=torch.float64).view(1, channel, 1, 1)
    else:
        dm = torch.tensor(getattr(consts, f'{args.dataset.lower()}_mean'), device=device, dtype=torch.float64).view(1, channel, 1, 1)
        ds = torch.tensor(getattr(consts, f'{args.dataset.lower()}_std'), device=device, dtype=torch.float64).view(1, channel, 1, 1)

    results = {rows: [] for rows in row_counts}

    for local_i, idx in enumerate(sample_indices):
        gt_data = tt(dst[idx][0]).double().to(device).unsqueeze(0)
        gt_label = torch.tensor([dst[idx][1]], dtype=torch.long, device=device)

        gt_data_norm = (gt_data - dm) / ds
        out = net(gt_data_norm)
        loss = criterion(out, gt_label)
        dy_dx = torch.autograd.grad(loss, net.parameters())
        original_dy_dx = [g.detach().clone() for g in dy_dx]

        grad_norm = sum(g.norm().item() for g in original_dy_dx if g is not None)
        has_nan = any(torch.isnan(g).any().item() for g in original_dy_dx if g is not None)
        has_inf = any(torch.isinf(g).any().item() for g in original_dy_dx if g is not None)
        print(f"  sample idx={idx}: grad_norm={grad_norm:.4f}  nan={has_nan}  inf={has_inf}", flush=True)

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
            _print_mask_debug(args, net, prefixes, prefix_layer_fracs,
                              original_dy_dx, keep_ids, entry_masks)

        sweep = compute_jacobian_rank_sweep(
            net=net,
            x_norm=gt_data_norm,
            y=gt_label,
            criterion=criterion,
            keep_ids=keep_ids,
            entry_masks=entry_masks,
            row_counts=row_counts,
            select_mode=args.jacobian_select_mode,
            qr_pivot=args.qr_pivot,
            device_for_J="cpu",
            print_svd_info=args.print_svd_info,
        )
        max_rows = max(row_counts)
        for rows in row_counts:
            jac_rank, jac_shape, used_rows, jac_unknowns = sweep[rows]
            results[rows].append(jac_rank)
            if rows == max_rows:
                print(f"    max_rows={rows}: rank={jac_rank}  shape={jac_shape}", flush=True)
            if progress_fn is not None:
                progress_fn(1)

        del gt_data, gt_label, gt_data_norm, out, loss, dy_dx, original_dy_dx
        if torch.cuda.is_available() and device.startswith("cuda"):
            torch.cuda.empty_cache()

    return results


def _mp_worker(rank, world_size, args, sample_chunks, row_counts, prefixes, prefix_layer_fracs,
               shared_results, progress_counter, progress_lock):
    """mp.spawn target: runs _worker_core on one GPU."""
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())

    if args.device.startswith("cuda") and torch.cuda.is_available():
        device = f"cuda:{rank % torch.cuda.device_count()}"
    else:
        device = "cpu"

    seed = args.run_id + 1 + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    def progress_fn(n):
        with progress_lock:
            progress_counter.value += n

    results = _worker_core(
        args, sample_chunks[rank], device, row_counts, prefixes, prefix_layer_fracs,
        progress_fn=progress_fn,
        debug=(rank == 0),
    )
    shared_results[rank] = results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--network", type=str, default="resnet18")
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--method", type=str, default="idlg", choices=["idlg", "masked"])
    parser.add_argument("--pretrained", action="store_true",
                        help="Load ImageNet-pretrained weights (affects normalization).")

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
                        help="Step between row counts. Generates range from unknowns to "
                             "--max_row_count in increments of stepsize. "
                             "Overrides --row_counts when set.")
    parser.add_argument("--max_row_count", type=int, default=None,
                        help="Upper bound for row counts when --stepsize is used.")
    parser.add_argument("--jacobian_select_mode", type=str, default="topk_abs",
                        choices=["topk_abs", "first", "random", "layer_spread"])
    parser.add_argument("--qr_pivot", action="store_true",
                        help="After building J with --jacobian_select_mode, reorder its rows "
                             "via QR column pivoting so each slice J[:k] contains the k most "
                             "linearly independent rows from the pool.")
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of parallel workers (1 = serial, 2-4 = one process per GPU).")
    parser.add_argument("--print_svd_info", action="store_true",
                        help="Print the 10 smallest singular values of the row-normalised "
                             "Jacobian at each row count.")

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

    if os.access('/work3/s234843/bachelor', os.R_OK | os.W_OK | os.X_OK):
        save_dir = '/work3/s234843/bachelor/results'
    else:
        save_dir = './results'
    os.makedirs(save_dir, exist_ok=True)

    dst, channel, num_classes, shape_img = load_dataset(args.dataset, _data_path())

    seed = args.run_id + 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    idx_shuffle = np.random.permutation(len(dst))
    sample_indices = idx_shuffle[:args.num_samples].tolist()

    unknowns = channel * shape_img[0] * shape_img[1]

    if args.stepsize is not None:
        row_counts = list(range(unknowns, args.max_row_count + 1, args.stepsize))
        if not row_counts or row_counts[-1] < args.max_row_count:
            row_counts.append(args.max_row_count)

    print(f"Unknowns: {unknowns}")
    print(f"Samples: {sample_indices}")

    total_steps = len(sample_indices) * len(row_counts)

    if args.num_workers == 1:
        device = args.device
        if args.device == "cuda":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif args.device.startswith("cuda") and not torch.cuda.is_available():
            print(f"CUDA requested via --device {args.device}, but CUDA is unavailable; using CPU.")
            device = "cpu"

        with tqdm(total=total_steps, desc="Jacobian rank") as pbar:
            rank_results = _worker_core(
                args, sample_indices, device, row_counts, prefixes, prefix_layer_fracs,
                progress_fn=lambda n: pbar.update(n),
                debug=True,
            )

    else:
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
        with tqdm(total=total_steps, desc="Jacobian rank") as pbar:
            while not spawn_ctx.join(timeout=0.2):
                current = progress_counter.value
                if current > last:
                    pbar.update(current - last)
                    last = current
            current = progress_counter.value
            if current > last:
                pbar.update(current - last)

        rank_results = {rows: [] for rows in row_counts}
        for worker_id in range(world_size):
            for rows in row_counts:
                rank_results[rows].extend(shared_results[worker_id][rows])

    xs, mean_ranks, std_ranks = [], [], []
    for rows in row_counts:
        xs.append(rows)
        mean_ranks.append(float(np.mean(rank_results[rows])))
        std_ranks.append(float(np.std(rank_results[rows])))

    print("\nPer-sample ranks:")
    for rows, mr, sr in zip(xs, mean_ranks, std_ranks):
        ranks = rank_results[rows]
        print(f"  rows={rows:6d}: {ranks}  mean={mr:.2f}  std={sr:.2f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = (
        f"jac_rank_{args.network}_{args.dataset}_{args.method}_"
        f"{args.jacobian_select_mode}_ns{args.num_samples}_{timestamp}"
    )

    csv_path = os.path.join(save_dir, base + ".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rows_used", "mean_rank", "std_rank",
                         "unknowns", "num_samples", "jacobian_select_mode", "per_sample_ranks"])
        for x, m, s in zip(xs, mean_ranks, std_ranks):
            ranks_str = ";".join(str(r) for r in rank_results[x])
            writer.writerow([x, m, s, unknowns, args.num_samples, args.jacobian_select_mode, ranks_str])

    legend_label = f"mean rank (select={args.jacobian_select_mode}, samples={args.num_samples}"
    if args.method == "masked":
        legend_label += f", mask={args.mask_mode}"
    legend_label += ")"

    plt.figure(figsize=(7, 5))
    plt.errorbar(xs, mean_ranks, yerr=std_ranks, marker="o", capsize=4, label=legend_label)
    plt.axhline(unknowns, linestyle="--", label=f"unknowns = {unknowns}")
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
