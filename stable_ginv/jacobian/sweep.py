import sys, os
import tempfile

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

from tqdm import tqdm

import functions.consts as consts
from functions.Dataset import load_dataset
from stable_ginv.io import resolve_storage_paths
from helper.Network import get_model, weights_init
from stable_ginv.masking import build_gradient_mask
from stable_ginv.metrics import compute_jacobian_rank_sweep


_TORCH_DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
}


def _get_last_fc_param_indices(net):
    """Return the parameter indices (into list(net.parameters())) of the last nn.Linear layer."""
    last_fc_name = None
    for name, module in net.named_modules():
        if isinstance(module, nn.Linear):
            last_fc_name = name
    if last_fc_name is None:
        return set()
    prefix = last_fc_name + "."
    return {i for i, (n, _) in enumerate(net.named_parameters())
            if n == last_fc_name or n.startswith(prefix)}


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
    dst, channel, num_classes, shape_img = load_dataset(args.dataset, resolve_storage_paths()[0])
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

        if args.exclude_fc:
            fc_ids = _get_last_fc_param_indices(net)
            keep_ids = set(keep_ids) - fc_ids
            if debug and local_i == 0:
                param_names = [n for n, _ in net.named_parameters()]
                excluded = sorted(fc_ids)
                tqdm.write(f"[exclude_fc] removed param indices {excluded}: "
                           f"{[param_names[i] for i in excluded]}", file=sys.stderr)

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
            independent=not args.no_independent,
            normalize_rows=args.normalise,
            force_fc=False,
            print_layer_dist=True,
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
