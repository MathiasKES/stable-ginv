import math

import numpy as np
import torch
import torch.nn.functional as F
import torch.autograd.forward_ad as fwAD
from skimage.metrics import structural_similarity as ssim

try:
    import scipy.linalg as _scipy_linalg
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

from functions.masking import flatten_observed_gradients


def compute_psnr_from_mse(mse: float, max_val: float = 1.0, eps: float = 1e-12) -> float:
    """Compute PSNR in dB from scalar MSE; returns inf when mse < eps."""
    mse = float(mse)
    if mse < eps:
        return float('inf')
    return 10.0 * math.log10((max_val * max_val) / mse)


def compute_ssim_batch(x, y):
    """Mean SSIM over a batch of [N, C, H, W] tensors in [0, 1]."""
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()

    scores = []
    for i in range(x_np.shape[0]):
        img_x = np.transpose(x_np[i], (1, 2, 0))
        img_y = np.transpose(y_np[i], (1, 2, 0))

        if img_x.shape[2] == 1:
            score = ssim(
                img_x.squeeze(-1),
                img_y.squeeze(-1),
                data_range=1.0,
                win_size=7
            )
        else:
            score = ssim(
                img_x,
                img_y,
                channel_axis=2,
                data_range=1.0,
                win_size=7
            )
        scores.append(score)

    return float(np.mean(scores))


def total_variation(x):
    """L1 total variation regulariser for a [N, C, H, W] tensor."""
    tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return tv_h + tv_w


def _get_layer_groups(net):
    """Map each parameter name to a group label for layer_spread.

    Top-level children become groups. A nn.Sequential is expanded one level
    when all of its direct children are leaf modules (no sub-children), e.g.
    VGG's features.0, features.2, ... Each ResNet BasicBlock/Bottleneck keeps
    its parent Sequential label (layer1, layer2, ...) because BasicBlocks have
    their own sub-modules.
    """
    groups = {}
    for child_name, child in net.named_children():
        if isinstance(child, torch.nn.Sequential):
            all_leaf = all(not list(gc.named_children()) for _, gc in child.named_children())
            if all_leaf:
                for gc_name, gc_module in child.named_children():
                    for param_name, _ in gc_module.named_parameters():
                        groups[f"{child_name}.{gc_name}.{param_name}"] = f"{child_name}.{gc_name}"
            else:
                for param_name, _ in child.named_parameters():
                    groups[f"{child_name}.{param_name}"] = child_name
        else:
            for param_name, _ in child.named_parameters():
                groups[f"{child_name}.{param_name}"] = child_name
    for param_name, _ in net.named_parameters(recurse=False):
        groups[param_name] = param_name
    return groups


def _layer_info(grads, keep_ids, entry_masks):
    """Return (sizes, ndims) for each tensor contributing to the flat gradient vector.

    ndim is the number of dimensions of the original parameter tensor:
      - ndim >= 2: weight tensors (conv, linear) — localized, informative for rank
      - ndim == 1: bias / BN scale / BN shift — small, diffuse, less informative
    """
    sizes, ndims = [], []
    for i, g in enumerate(grads):
        if g is None:
            continue
        if entry_masks is not None:
            m = entry_masks[i]
            if m is None:
                continue
            sizes.append(int(m.reshape(-1).sum().item()))
        else:
            if keep_ids is not None and i not in keep_ids:
                continue
            sizes.append(g.numel())
        ndims.append(g.ndim)
    return sizes, ndims


def _build_jacobian(net, x_norm, y, criterion, keep_ids, entry_masks,
                    max_entries, select_mode, device_for_J, progress_fn=None):
    """Forward pass + row selection + J construction.

    Automatically chooses the cheaper direction:
      - Forward-mode AD (column-wise, unknowns passes) when unknowns < used_entries.
      - Backward-mode AD (row-wise, used_entries passes) otherwise.

    Returns (J, total_entries, unknowns).
    J has shape (used_entries, unknowns).
    """
    params = tuple(net.parameters())
    x_norm = x_norm.detach()
    unknowns = x_norm.numel()

    # One backward pass (no create_graph) to get g_obs magnitudes for row selection.
    x_sel = x_norm.clone().requires_grad_(True)
    out = net(x_sel)
    loss = criterion(out, y)
    grads = torch.autograd.grad(loss, params, create_graph=False)
    g_obs_det = flatten_observed_gradients(grads, keep_ids=keep_ids, entry_masks=entry_masks).detach()
    total_entries = g_obs_det.numel()

    if max_entries is None or max_entries >= total_entries:
        selected_idx = torch.arange(total_entries, device=g_obs_det.device)
    else:
        k = int(max_entries)
        if k <= 0:
            raise ValueError("max_entries must be positive")
        if select_mode == "topk_abs":
            selected_idx = torch.topk(g_obs_det.abs(), k=k, largest=True).indices
        elif select_mode == "first":
            selected_idx = torch.arange(k, device=g_obs_det.device)
        elif select_mode == "random":
            selected_idx = torch.randperm(total_entries, device=g_obs_det.device)[:k]
        elif select_mode == "layer_spread":
            # Spread budget evenly across layer groups determined by _get_layer_groups.
            # For architectures like ResNet, each top-level child (conv1, bn1, layer1, ...)
            # is its own group. For architectures like VGG where top-level children are
            # nn.Sequential containers of simple layers (Conv2d, ReLU, ...), each direct
            # child of the Sequential becomes its own group (features.0, features.2, ...),
            # giving per-conv-layer granularity. Within each group, the top-magnitude
            # entries are selected globally across all tensors in that group.
            layer_sizes, _ = _layer_info(grads, keep_ids, entry_masks)
            named_params = list(net.named_parameters())
            param_to_group = _get_layer_groups(net)

            # Build group label for each observed tensor (same filtering as _layer_info)
            group_of = []
            for i, (name, _p) in enumerate(named_params):
                if grads[i] is None:
                    continue
                if entry_masks is not None:
                    if entry_masks[i] is None:
                        continue
                else:
                    if keep_ids is not None and i not in keep_ids:
                        continue
                group_of.append(param_to_group.get(name, name.split('.')[0]))

            # Ordered unique groups (first-occurrence order = network depth order)
            seen_grps = set()
            groups = []
            for grp in group_of:
                if grp not in seen_grps:
                    groups.append(grp)
                    seen_grps.add(grp)

            num_groups = len(groups)
            group_totals = {grp: 0 for grp in groups}
            group_slices = {grp: [] for grp in groups}
            offset = 0
            for sz, grp in zip(layer_sizes, group_of):
                group_totals[grp] += sz
                group_slices[grp].append((offset, sz))
                offset += sz

            base = k // num_groups
            remainder = k % num_groups
            order = sorted(groups, key=lambda grp: group_totals[grp], reverse=True)
            per_group = {grp: base for grp in groups}
            for i in range(remainder):
                per_group[order[i]] += 1
            per_group = {grp: min(per_group[grp], group_totals[grp]) for grp in groups}

            # Redistribute surplus from groups smaller than their allocation (e.g. bn1
            # in ResNet-18 has 128 entries but would otherwise waste ~1300 budget slots).
            leftover = k - sum(per_group.values())
            if leftover > 0:
                for grp in order:
                    if per_group[grp] < group_totals[grp]:
                        give = min(leftover, group_totals[grp] - per_group[grp])
                        per_group[grp] += give
                        leftover -= give
                        if leftover == 0:
                            break

            indices = []
            for grp in groups:
                n = per_group[grp]
                slices = group_slices[grp]
                if n <= 0 or not slices:
                    continue
                if n >= group_totals[grp]:
                    for off, sz in slices:
                        indices.append(torch.arange(sz, device=g_obs_det.device) + off)
                else:
                    grp_vals = torch.cat([g_obs_det[off:off + sz] for off, sz in slices])
                    top_local = torch.topk(grp_vals.abs(), k=n, largest=True).indices
                    local_to_global = torch.cat([
                        torch.arange(sz, device=g_obs_det.device) + off
                        for off, sz in slices
                    ])
                    indices.append(local_to_global[top_local])

            selected_idx = (torch.cat(indices) if indices
                            else torch.arange(min(k, total_entries), device=g_obs_det.device))
        else:
            raise ValueError(f"Unknown select_mode: {select_mode}")

    used_entries = selected_idx.numel()
    selected_idx_for_grad = selected_idx.to(x_norm.device)
    J = torch.empty((used_entries, unknowns), dtype=x_norm.dtype, device=device_for_J)

    if unknowns < used_entries:
        # Forward-mode: one pass per pixel column — ~10x faster when unknowns << max_entries.
        # Requires PyTorch >= 2.0 for autograd.grad to propagate dual tangents.
        for j in range(unknowns):
            tangent = torch.zeros_like(x_norm)
            tangent.reshape(-1)[j] = 1.0
            with fwAD.dual_level():
                x_dual = fwAD.make_dual(x_norm, tangent)
                out_d = net(x_dual)
                loss_d = criterion(out_d, y)
                grads_d = torch.autograd.grad(loss_d, params)
                g_obs_d = flatten_observed_gradients(grads_d, keep_ids=keep_ids, entry_masks=entry_masks)
                g_col = g_obs_d[selected_idx_for_grad]
                col = fwAD.unpack_dual(g_col).tangent
                if col is None:
                    raise RuntimeError(
                        "Forward-mode AD did not propagate through autograd.grad. "
                        "Requires PyTorch >= 2.0. Upgrade PyTorch or open an issue."
                    )
                J[:, j] = col.detach().reshape(-1).to(device_for_J)
            if progress_fn is not None:
                progress_fn(1)
    else:
        # Backward-mode: one pass per gradient row.
        x_req = x_norm.requires_grad_(True)
        out = net(x_req)
        loss = criterion(out, y)
        grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
        g_obs_full = flatten_observed_gradients(grads, keep_ids=keep_ids, entry_masks=entry_masks)
        g_sel = g_obs_full[selected_idx]

        for i in range(used_entries):
            grad_i = torch.autograd.grad(
                g_sel[i], x_req,
                retain_graph=(i < used_entries - 1),
                create_graph=False,
                allow_unused=False,
            )[0]
            J[i] = grad_i.reshape(-1).detach().to(device_for_J)
            if progress_fn is not None:
                progress_fn(1)

    return J, total_entries, unknowns


def _rank_of_J(J, print_svd_info=False, normalize_rows=True):
    """Compute numerical rank of J via matrix_rank (runs on CPU/LAPACK).

    normalize_rows=True (default): each row is divided by its L2 norm before
        SVD. Stabilises the threshold when rows span large magnitude ranges.
    normalize_rows=False: raw J is passed to matrix_rank. The default relative
        threshold (max(M,N)*eps*sigma_max) then reflects actual gradient magnitudes.
    """
    J_cpu = J.cpu()
    if normalize_rows:
        row_norms = torch.norm(J_cpu, dim=1, keepdim=True).clamp_min(1e-30)
        J_for_rank = J_cpu / row_norms
    else:
        J_for_rank = J_cpu
    if print_svd_info:
        svd_vals = torch.linalg.svdvals(J_for_rank)
        M, N = J_for_rank.shape
        sigma_max = svd_vals[0].item()
        eps = torch.finfo(J_for_rank.dtype).eps
        atol = max(M, N) * eps * sigma_max
        bottom = svd_vals[-10:].tolist()
        print(f"  [SVD] M={M} N={N}  normalised={normalize_rows}  "
              f"sigma_max={sigma_max:.3e}  atol={atol:.3e}  "
              f"bottom-10: {[f'{v:.3e}' for v in bottom]}", flush=True)
    return int(torch.linalg.matrix_rank(J_for_rank).item())


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
    print_svd_info=False,
):
    """Rank of J = d vec(g_obs) / d vec(x), built row-by-row to avoid OOM.

    print_svd_info: if True, print the 10 smallest singular values of the
                    row-normalised Jacobian to help diagnose rank behaviour.
    """
    J, total_entries, unknowns = _build_jacobian(
        net, x_norm, y, criterion, keep_ids, entry_masks, max_entries, select_mode, device_for_J,
    )
    jac_rank = _rank_of_J(J, print_svd_info=print_svd_info)
    return jac_rank, tuple(J.shape), J.shape[0], unknowns


def _qr_pivot_rows(J):
    """Return J reordered by QR column pivoting on J^T."""
    if not _SCIPY_AVAILABLE:
        raise ImportError("qr_pivot requires scipy. Install with: pip install scipy")
    _, _, pivots = _scipy_linalg.qr(J.cpu().numpy().T, pivoting=True, mode="economic")
    return J[torch.from_numpy(pivots.astype(np.int64))]


def compute_jacobian_rank_sweep(
    net,
    x_norm,
    y,
    criterion,
    keep_ids=None,
    entry_masks=None,
    row_counts=None,
    select_mode="topk_abs",
    qr_pivot=False,
    device_for_J="cpu",
    print_svd_info=False,
    j_progress_fn=None,
    rank_progress_fn=None,
    independent=False,
    normalize_rows=True,
):
    """Rank of J for multiple row counts.

    independent=False (default):
        Builds J once at max(row_counts) then slices J_max[:k] for each k.
        Fast (one J build per sample) but rank at k depends on max_row_count
        because the pool composition changes with the budget. Correct for
        topk_abs; misleading for layer_spread.

    independent=True:
        Builds J independently for each k using max_entries=k. Rank at k
        reflects exactly the information available when the attacker receives
        k gradient entries selected by select_mode. Theoretically correct for
        all select modes. Costs len(row_counts) × more forward passes.

    qr_pivot in independent mode: J_k is already the full k-entry selection,
        so QR-pivoting it and taking all k rows gives the same rank as J_k.
        qr_pivot is most useful in the default (pool-slice) mode where it
        finds the best k rows from a larger pool.

    Returns ({k: (rank, shape, used_entries, unknowns)}, results_qr_or_None).
    """
    if not row_counts:
        raise ValueError("row_counts must be a non-empty list")

    if independent:
        results = {}
        results_qr = {} if qr_pivot else None
        for k in row_counts:
            J_k, _, unknowns = _build_jacobian(
                net, x_norm, y, criterion, keep_ids, entry_masks,
                k, select_mode, device_for_J, progress_fn=j_progress_fn,
            )
            rank = _rank_of_J(J_k, print_svd_info=print_svd_info, normalize_rows=normalize_rows)
            results[k] = (rank, tuple(J_k.shape), J_k.shape[0], unknowns)
            if rank_progress_fn is not None:
                rank_progress_fn(1)
            if qr_pivot:
                # QR reordering all rows of an independently built J_k cannot
                # change rank, so avoid the expensive QR + second SVD.
                results_qr[k] = (rank, tuple(J_k.shape), J_k.shape[0], unknowns)
                if rank_progress_fn is not None:
                    rank_progress_fn(1)
            del J_k
        return results, results_qr

    # --- build-once (pool-slice) mode ---
    J_max, total_entries, unknowns = _build_jacobian(
        net, x_norm, y, criterion, keep_ids, entry_masks, max(row_counts), select_mode, device_for_J,
        progress_fn=j_progress_fn,
    )

    results = {}
    for k in row_counts:
        J_k = J_max[:k]
        rank = _rank_of_J(J_k, print_svd_info=print_svd_info, normalize_rows=normalize_rows)
        results[k] = (rank, tuple(J_k.shape), J_k.shape[0], unknowns)
        if rank_progress_fn is not None:
            rank_progress_fn(1)

    if not qr_pivot:
        return results, None

    # QR with column pivoting on J^T: pivots[i] is the i-th most linearly independent row of J.
    J_max_qr = _qr_pivot_rows(J_max)
    results_qr = {}
    for k in row_counts:
        J_k = J_max_qr[:k]
        rank = _rank_of_J(J_k, print_svd_info=False, normalize_rows=normalize_rows)
        results_qr[k] = (rank, tuple(J_k.shape), J_k.shape[0], unknowns)
        if rank_progress_fn is not None:
            rank_progress_fn(1)

    return results, results_qr


def compute_grad_match_loss(dummy_dy_dx, selected_original, selected_entry_masks=None, grad_loss="cos", eps=1e-12):
    """Gradient matching loss between dummy and original gradients (l2 or cosine)."""
    grad_loss = grad_loss.lower()

    if grad_loss not in {"l2", "cos"}:
        raise ValueError(f"Unknown grad_loss: {grad_loss}")

    if grad_loss == "l2":
        total = 0.0
        num_terms = 0

        if selected_entry_masks is not None:
            for gx, gy, m in zip(dummy_dy_dx, selected_original, selected_entry_masks):
                gx_sel = gx[m]
                gy_sel = gy[m]
                if gx_sel.numel() == 0:
                    continue
                diff = gx_sel - gy_sel
                total = total + (diff ** 2).sum()
                num_terms += diff.numel()
        else:
            for gx, gy in zip(dummy_dy_dx, selected_original):
                diff = gx - gy
                total = total + (diff ** 2).sum()
                num_terms += diff.numel()

        return total, num_terms

    elif grad_loss == "cos":
        gx_all = []
        gy_all = []

        if selected_entry_masks is not None:
            for gx, gy, m in zip(dummy_dy_dx, selected_original, selected_entry_masks):
                gx_sel = gx[m]
                gy_sel = gy[m]
                if gx_sel.numel() == 0:
                    continue
                gx_all.append(gx_sel.reshape(-1))
                gy_all.append(gy_sel.reshape(-1))
        else:
            for gx, gy in zip(dummy_dy_dx, selected_original):
                gx_all.append(gx.reshape(-1))
                gy_all.append(gy.reshape(-1))

        if len(gx_all) == 0:
            raise ValueError("No gradient entries selected for cosine loss.")

        gx_cat = torch.cat(gx_all)
        gy_cat = torch.cat(gy_all)

        cos_sim = F.cosine_similarity(
            gx_cat.unsqueeze(0),
            gy_cat.unsqueeze(0),
            dim=1,
            eps=eps
        )

        return 1.0 - cos_sim[0], gx_cat.numel()
