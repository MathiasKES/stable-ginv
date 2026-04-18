import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import math
from Network import LeNet, LeNet_bigger, MediumCNN, BiggerCNN, get_model
#from skimage.metrics import structural_similarity as ssim
import torch
import torch.nn as nn
import numpy as np

def flatten_observed_gradients(grad_list, keep_ids=None, entry_masks=None):
    """
    Flatten observed gradients.
    Two modes:
    - tensor-wise masking via keep_ids
    - elementwise masking via entry_masks
    """
    flat = []
    for i, g in enumerate(grad_list):
        if g is None:
            continue
        if entry_masks is not None:
            m = entry_masks[i]
            if m is None:
                continue
            flat.append(g.reshape(-1)[m.reshape(-1)])
        else:
            if keep_ids is not None and i not in keep_ids:
                continue
            flat.append(g.reshape(-1))
    if len(flat) == 0:
        raise ValueError("No observed gradients selected.")
    return torch.cat(flat, dim=0)

def compute_jacobian_rank(
    net,
    x_norm,
    y,
    criterion,
    keep_ids=None,
    entry_masks=None,
    max_entries=None,
    select_mode="topk_abs",
    device_for_J="cpu",
    rank_tol=1e-6,
    normalize_rows=False,
):
    """
    Compute rank of J = d vec(g_obs(x)) / d vec(x),
    using only a subset of observed gradient entries.

    This version computes the Jacobian row-by-row to avoid OOM.
    """
    params = tuple(net.parameters())
    x_norm = x_norm.detach().clone().requires_grad_(True)

    # Build observed gradient vector once
    out = net(x_norm)
    loss = criterion(out, y)
    grads = torch.autograd.grad(
        loss,
        params,
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )
    g_obs = flatten_observed_gradients(grads, keep_ids=keep_ids, entry_masks=entry_masks)

    total_entries = g_obs.numel()

    if max_entries is None or max_entries >= total_entries:
        selected_idx = torch.arange(total_entries, device=g_obs.device)
    else:
        k = int(max_entries)
        if k <= 0:
            raise ValueError("max_entries must be positive")

        if select_mode == "topk_abs":
            selected_idx = torch.topk(g_obs.detach().abs(), k=k, largest=True).indices
        elif select_mode == "first":
            selected_idx = torch.arange(k, device=g_obs.device)
        elif select_mode == "random":
            selected_idx = torch.randperm(total_entries, device=g_obs.device)[:k]
        else:
            raise ValueError(f"Unknown select_mode: {select_mode}")

    g_sel = g_obs[selected_idx]

    used_entries = g_sel.numel()
    unknowns = x_norm.numel()

    J = torch.empty((used_entries, unknowns), dtype=x_norm.dtype, device=device_for_J)
    for i in range(used_entries):
        grad_i = torch.autograd.grad(
            g_sel[i],
            x_norm,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]
        J[i] = grad_i.reshape(-1).detach().to(device_for_J)

    if normalize_rows:
        row_norms = torch.norm(J, dim=1, keepdim=True).clamp_min(1e-12)
        J = J / row_norms

    jac_rank = int(torch.linalg.matrix_rank(J).item())

    # print(f"jacobian rank: {jac_rank}")
    # print(f"matrix shape: {tuple(J.shape)}")
    # print(f"max possible rank: {min(J.shape)}")
    # print(f"rank deficiency: {min(J.shape) - jac_rank}")

    return jac_rank, tuple(J.shape), used_entries, unknowns

def get_entry_masks_by_gradsize(original_dy_dx, mode="topk_entries", topk=None, top_frac=None, candidate_ids=None):
    """
    Elementwise masking:
    - topk_entries: keep exactly topk scalar gradient entries by absolute magnitude
    - topfrac_entries: keep exactly round(top_frac * total_entries) scalar entries

    Returns:
        entry_masks: list of boolean tensors matching original_dy_dx
        kept_entries: exact number of kept scalar entries
        total_entries: total number of scalar entries
    """
    pieces = []
    meta = []

    for i, g in enumerate(original_dy_dx):
        if candidate_ids is not None and i not in candidate_ids:
            meta.append((i, 0, None))
            continue
        if g is None:
            meta.append((i, 0, None))
            continue
        flat = g.detach().abs().reshape(-1)
        pieces.append(flat)
        meta.append((i, flat.numel(), g.shape))

    if len(pieces) == 0:
        raise ValueError("No gradients available for masking.")

    all_vals = torch.cat(pieces, dim=0)
    total_entries = all_vals.numel()

    if mode == "topk_entries":
        if topk is None:
            raise ValueError("topk must be set for mode='topk_entries'")
        k = max(1, min(int(topk), total_entries))
    elif mode == "topfrac_entries":
        if top_frac is None:
            raise ValueError("top_frac must be set for mode='topfrac_entries'")
        k = max(1, min(int(round(top_frac * total_entries)), total_entries))
    else:
        raise ValueError(f"Unknown entrywise mode: {mode}")

    top_idx = torch.topk(all_vals, k=k, largest=True).indices
    global_mask = torch.zeros(total_entries, dtype=torch.bool, device=all_vals.device)
    global_mask[top_idx] = True

    entry_masks = []
    offset = 0
    for i, numel, shape in meta:
        g = original_dy_dx[i]

        if g is None or shape is None or numel == 0:
            entry_masks.append(None)
            continue

        local_mask = global_mask[offset:offset + numel].reshape(shape)
        entry_masks.append(local_mask)
        offset += numel

    return entry_masks, k, total_entries

def get_entry_masks_by_prefix_group(
    net,
    original_dy_dx,
    prefixes,
    mode="topfrac_entries",
    topk=None,
    top_frac=None,
    prefix_top_fracs=None
):
    """
    Elementwise masking applied per prefix group.

    For each prefix:
      - collect all gradient entries from parameters whose name starts with that prefix
      - rank entries only within that prefix group
      - keep top-k or top-fraction within that group

    Returns:
        entry_masks: list of boolean tensors matching original_dy_dx
        kept_entries: total kept scalar entries across all selected prefixes
        total_entries: total scalar entries across all selected prefixes
    """
    named_params = list(net.named_parameters())
    entry_masks = [None] * len(original_dy_dx)

    kept_entries = 0
    total_entries = 0

    if prefix_top_fracs is None:
        prefix_top_fracs = {}

    for prefix in prefixes:
        group_infos = []
        pieces = []

        for i, (name, _) in enumerate(named_params):
            if not name.startswith(prefix):
                continue

            g = original_dy_dx[i]
            if g is None:
                continue

            flat = g.detach().abs().reshape(-1)
            group_infos.append((i, g.shape, flat.numel()))
            pieces.append(flat)

        if len(pieces) == 0:
            continue

        all_vals = torch.cat(pieces, dim=0)
        n = all_vals.numel()
        total_entries += n

        if mode == "topk_entries":
            if topk is None:
                raise ValueError("topk must be set for mode='topk_entries'")
            k = max(1, min(int(topk), n))
        elif mode == "topfrac_entries":
            local_top_frac = prefix_top_fracs.get(prefix, top_frac)
            if local_top_frac is None:
                raise ValueError("top_frac must be set for mode='topfrac_entries'")
            k = max(1, min(int(round(local_top_frac * n)), n))
        else:
            raise ValueError(f"Unknown mode: {mode}")

        top_idx = torch.topk(all_vals, k=k, largest=True).indices
        group_mask = torch.zeros(n, dtype=torch.bool, device=all_vals.device)
        group_mask[top_idx] = True

        offset = 0
        for i, shape, numel in group_infos:
            local_mask = group_mask[offset:offset + numel].reshape(shape)
            entry_masks[i] = local_mask
            offset += numel

        kept_entries += k

    if kept_entries == 0:
        raise ValueError(f"No gradients matched prefixes={prefixes}")

    return entry_masks, kept_entries, total_entries

def get_keep_ids_by_gradsize(original_dy_dx, mode="topk", topk=10, top_frac=None, threshold=None, metric="l2", candidate_ids=None):
    """
    Tensor-wise masking.
    """
    sizes = []
    for i, g in enumerate(original_dy_dx):
        if candidate_ids is not None and i not in candidate_ids:
            continue
        if g is None:
            sizes.append((i, float("-inf")))
            continue

        if metric == "l2":
            s = g.detach().norm(p=2).item()
        elif metric == "mean_abs":
            s = g.detach().abs().mean().item()
        elif metric == "sum_abs":
            s = g.detach().abs().sum().item()
        else:
            raise ValueError(f"Unknown metric: {metric}")

        sizes.append((i, s))

    sizes_sorted = sorted(sizes, key=lambda x: x[1], reverse=True)

    if mode == "topk":
        k = min(topk, len(sizes_sorted))
        keep = [i for i, _ in sizes_sorted[:k]]

    elif mode == "topfrac":
        if top_frac is None:
            raise ValueError("top_frac must be set for mode='topfrac'")
        k = max(1, int(round(top_frac * len(sizes_sorted))))
        keep = [i for i, _ in sizes_sorted[:k]]

    elif mode == "threshold":
        if threshold is None:
            raise ValueError("threshold must be set for mode='threshold'")
        keep = [i for i, s in sizes_sorted if s >= threshold]
        if len(keep) == 0:
            keep = [sizes_sorted[0][0]]

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return sorted(keep), sizes_sorted

def build_network(name: str, channel: int, num_classes: int, input_size):
    if name == "LeNet":
        return LeNet(channel=channel, num_classes=num_classes, input_size=input_size)
    if name == "LeNet_bigger":
        return LeNet_bigger(channel=channel, num_classes=num_classes, input_size=input_size)
    if name == "MediumCNN":
        return MediumCNN(channel=channel, num_classes=num_classes, input_size=input_size)
    if name == "BiggerCNN":
        return BiggerCNN(channel=channel, num_classes=num_classes, input_size=input_size)
    if name.lower().startswith("resnet") or name.lower().startswith("wide_resnet"):
        return get_model(
            network=name.lower(),
            channel=channel,
            num_classes=num_classes,
            input_size=input_size,
        )
    raise ValueError(f"Unknown NETWORK_NAME: {name}")

def get_prefix_keep_ids(net, prefixes):
    keep = set()
    for idx, (name, _) in enumerate(net.named_parameters()):
        if any(name.startswith(p) for p in prefixes):
            keep.add(idx)
    if len(keep) == 0:
        raise ValueError(f"No parameters matched prefixes={prefixes}")
    return keep

def get_keep_ids(mask_mode: str, net=None, prefixes=None):
    """
    Returns a set of parameter indices to keep.

    - If prefixes is provided: uses net.named_parameters() and keeps any parameter whose
      name starts with one of the prefixes.
    - Otherwise uses your legacy index-based masks (works for LeNet/MediumCNN etc.).

    Example:
      keep_ids = get_keep_ids(mask_mode="prefix", net=net, prefixes=("conv1","layer1","linear"))
    """

    # --- prefix-based mode (model-agnostic) ---
    if prefixes is not None:
        if net is None:
            raise ValueError("prefix-based keep_ids requires 'net'")
        keep = set()
        for idx, (name, _) in enumerate(net.named_parameters()):
            if any(name.startswith(p) for p in prefixes):
                keep.add(idx)
        if len(keep) == 0:
            raise ValueError(f"No parameters matched prefixes={prefixes}")
        return keep
    
    if mask_mode == "all":
        if net is None:
            return set(range(8))  # legacy behavior
        return set(range(len(list(net.parameters()))))  # works for any net if provided

    raise ValueError(f"Unknown MASK_MODE: {mask_mode}")

def compute_psnr_from_mse(mse: float, max_val: float = 1.0, eps: float = 1e-12) -> float:
    """Compute PSNR in dB from a scalar MSE. Assumes images are in [0, max_val]."""
    mse = float(mse)
    if mse < eps:
        return float('inf')
    return 10.0 * math.log10((max_val * max_val) / (mse + eps))

# def compute_ssim_batch(x, y):
#     """
#     x, y: torch tensors of shape [N, C, H, W] in [0, 1]
#     Returns mean SSIM over the batch.
#     """
#     x_np = x.detach().cpu().numpy()
#     y_np = y.detach().cpu().numpy()

#     scores = []
#     for i in range(x_np.shape[0]):
#         img_x = np.transpose(x_np[i], (1, 2, 0))  # C,H,W -> H,W,C
#         img_y = np.transpose(y_np[i], (1, 2, 0))

#         if img_x.shape[2] == 1:
#             score = ssim(
#                 img_x.squeeze(-1),
#                 img_y.squeeze(-1),
#                 data_range=1.0,
#                 win_size=7
#             )
#         else:
#             score = ssim(
#                 img_x,
#                 img_y,
#                 channel_axis=2,
#                 data_range=1.0,
#                 win_size=7
#             )
#         scores.append(score)

#     return float(np.mean(scores))

def save_recon_panel(params: dict, panel_gt_pil, panel_idlg_pil, panel_masked_pil,
                     save_dir, block_idx, dataset, mask_desc: str, timestamp_str: str,
                     methods: str = "both"):
    """
    Save a panel with dynamic rows depending on methods:

    methods="both"   -> rows: GT, iDLG, iDLG_masked
    methods="idlg"   -> rows: GT, iDLG
    methods="masked" -> rows: GT, iDLG_masked
    """
    n = len(panel_gt_pil)
    if n == 0:
        return

    rows = [("GT", panel_gt_pil)]

    if methods in ["idlg", "both"]:
        rows.append(("iDLG", panel_idlg_pil))

    if methods in ["masked", "both"]:
        rows.append(("iDLG_masked", panel_masked_pil))

    num_rows = len(rows)
    fig = plt.figure(figsize=(2.2 * n + 1.6, 2.1 * num_rows))

    # Row labels + images
    for r, (row_name, row_imgs) in enumerate(rows):
        # y-position for label from top to bottom
        y = 1.0 - (r + 0.5) / num_rows
        fig.text(0.01, y, row_name, va='center', ha='left',
                 fontsize=14, fontweight='bold')

        for j in range(n):
            ax = plt.subplot(num_rows, n, r * n + 1 + j)
            ax.imshow(row_imgs[j], cmap='gray' if dataset == 'MNIST' else None)
            if r == 0:
                ax.set_title(f"exp {j}")
            ax.axis('off')

    plt.tight_layout(rect=(0.08, 0.0, 1.0, 1.0))
    job_id = os.environ.get("LSB_JOBID", "")
    prefix = f"{timestamp_str}_job{job_id}" if job_id else timestamp_str
    out_path = os.path.join(
        save_dir,
        f"{prefix}_{'_'.join(f'{key}{val}' for key, val in params.items())}_{mask_desc}_{methods}.png"
    )
    plt.savefig(out_path, dpi=250, bbox_inches='tight')
    plt.close(fig)

    try:
        os.chmod(out_path, 0o770)
    except Exception as e:
        print(f"Warning: failed to set permissions for {out_path}: {e}")

    print("Saved reconstruction panel to:", out_path)
    return out_path

def build_gradient_mask(
    method,
    mask_mode,
    net,
    original_dy_dx,
    prefixes=(),
    prefix_layer_fracs=None,
    gradsize_topk=20,
    gradsize_topfrac=0.5,
    gradsize_threshold=None,
    gradsize_metric="l2",
):
    candidate_ids = None
    keep_ids = None
    entry_masks = None

    if prefix_layer_fracs is None:
        prefix_layer_fracs = {}

    if method == "idlg":
        keep_ids = get_keep_ids("all", net=net)
        return keep_ids, entry_masks

    if mask_mode.startswith("prefix_"):
        candidate_ids = get_prefix_keep_ids(net, prefixes)

    if mask_mode == "gradsize_topk":
        keep_ids, _ = get_keep_ids_by_gradsize(
            original_dy_dx, mode="topk", topk=gradsize_topk, metric=gradsize_metric
        )
    elif mask_mode == "gradsize_topfrac":
        keep_ids, _ = get_keep_ids_by_gradsize(
            original_dy_dx, mode="topfrac", top_frac=gradsize_topfrac, metric=gradsize_metric
        )
    elif mask_mode == "gradsize_topk_entries":
        entry_masks, _, _ = get_entry_masks_by_gradsize(
            original_dy_dx, mode="topk_entries", topk=gradsize_topk
        )
    elif mask_mode == "gradsize_topfrac_entries":
        entry_masks, _, _ = get_entry_masks_by_gradsize(
            original_dy_dx, mode="topfrac_entries", top_frac=gradsize_topfrac
        )
    elif mask_mode == "prefix_topk":
        keep_ids, _ = get_keep_ids_by_prefix_group(
            net=net,
            original_dy_dx=original_dy_dx,
            prefixes=prefixes,
            mode="topk",
            topk=gradsize_topk,
            prefix_top_ks={k: int(v) for k, v in prefix_layer_fracs.items()},
            metric=gradsize_metric,
        )
    elif mask_mode == "prefix_topfrac":
        keep_ids, _ = get_keep_ids_by_gradsize(
            original_dy_dx, mode="topfrac", top_frac=gradsize_topfrac,
            metric=gradsize_metric, candidate_ids=candidate_ids
        )
    elif mask_mode == "prefix_topk_entries":
        entry_masks, _, _ = get_entry_masks_by_gradsize(
            original_dy_dx, mode="topk_entries", topk=gradsize_topk,
            candidate_ids=candidate_ids
        )
    elif mask_mode == "prefix_topfrac_entries":
        entry_masks, _, _ = get_entry_masks_by_gradsize(
            original_dy_dx, mode="topfrac_entries", top_frac=gradsize_topfrac,
            candidate_ids=candidate_ids
        )
    elif mask_mode == "prefix_topk_entries_layer":
        entry_masks, _, _ = get_entry_masks_by_prefix_group(
            net=net,
            original_dy_dx=original_dy_dx,
            prefixes=prefixes,
            mode="topk_entries",
            topk=gradsize_topk,
            prefix_top_fracs=prefix_layer_fracs,
        )
    elif mask_mode == "prefix_topfrac_entries_layer":
        entry_masks, _, _ = get_entry_masks_by_prefix_group(
            net=net,
            original_dy_dx=original_dy_dx,
            prefixes=prefixes,
            mode="topfrac_entries",
            top_frac=gradsize_topfrac,
        )
    elif mask_mode == "gradsize_threshold":
        keep_ids, _ = get_keep_ids_by_gradsize(
            original_dy_dx, mode="threshold", threshold=gradsize_threshold,
            metric=gradsize_metric
        )
    elif mask_mode == "prefix":
        keep_ids = get_keep_ids(mask_mode="prefix", net=net, prefixes=prefixes)
    else:
        keep_ids = get_keep_ids(mask_mode, net=net)

    return keep_ids, entry_masks

def total_variation(x):
    tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return tv_h + tv_w

def get_keep_ids_by_prefix_group(
    net,
    original_dy_dx,
    prefixes,
    mode="topfrac",
    topk=None,
    top_frac=None,
    prefix_top_fracs=None,
    prefix_top_ks=None,
    metric="l2",
):
    """
    Tensor-wise masking applied per prefix group.

    For each prefix:
      - collect parameter tensors whose name starts with that prefix
      - rank tensors within that prefix group by gradient size
      - keep top-k or top-fraction within that group

    Returns:
        keep_ids: sorted list of kept parameter indices
        ranked_by_prefix: dict mapping prefix -> sorted [(idx, score), ...]
    """
    if prefix_top_fracs is None:
        prefix_top_fracs = {}
    
    if prefix_top_ks is None:
        prefix_top_ks = {}

    named_params = list(net.named_parameters())
    keep = set()
    ranked_by_prefix = {}

    for prefix in prefixes:
        sizes = []

        for i, (name, _) in enumerate(named_params):
            if not name.startswith(prefix):
                continue

            g = original_dy_dx[i]
            if g is None:
                continue

            if metric == "l2":
                s = g.detach().norm(p=2).item()
            elif metric == "mean_abs":
                s = g.detach().abs().mean().item()
            elif metric == "sum_abs":
                s = g.detach().abs().sum().item()
            else:
                raise ValueError(f"Unknown metric: {metric}")

            sizes.append((i, s))

        if len(sizes) == 0:
            continue

        sizes_sorted = sorted(sizes, key=lambda x: x[1], reverse=True)
        ranked_by_prefix[prefix] = sizes_sorted

        if mode == "topk":
            local_topk = prefix_top_ks.get(prefix, topk)
            if local_topk is None:
                raise ValueError("topk must be set for mode='topk'")
            k = max(0, min(int(local_topk), len(sizes_sorted)))

        elif mode == "topfrac":
            local_top_frac = prefix_top_fracs.get(prefix, top_frac)
            if local_top_frac is None:
                raise ValueError("top_frac must be set for mode='topfrac'")
            k = max(1, min(int(round(local_top_frac * len(sizes_sorted))), len(sizes_sorted)))

        else:
            raise ValueError(f"Unknown mode: {mode}")

        if k > 0:
            keep.update(i for i, _ in sizes_sorted[:k])

    if len(keep) == 0:
        raise ValueError(f"No parameters matched prefixes={prefixes}")

    return sorted(keep), ranked_by_prefix