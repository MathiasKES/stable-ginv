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


def _layer_boundaries(grads, keep_ids, entry_masks):
    """Return the number of entries each tensor contributes to the flat gradient vector."""
    sizes = []
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
    return sizes


def _build_jacobian(net, x_norm, y, criterion, keep_ids, entry_masks,
                    max_entries, select_mode, device_for_J):
    """Forward pass + row selection + J construction.

    Automatically chooses the cheaper direction:
      - Forward-mode AD (column-wise, unknowns passes) when unknowns < used_entries.
      - Backward-mode AD (row-wise, used_entries passes) otherwise.

    Returns (J, total_entries, unknowns).
    J has shape (used_entries, unknowns).
    """
    params = tuple(net.parameters())
    x_norm = x_norm.detach().clone()
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
            # Distribute k budget evenly across layers; within each layer take top-abs entries.
            layer_sizes = _layer_boundaries(grads, keep_ids, entry_masks)
            num_layers = len(layer_sizes)
            base = k // num_layers
            remainder = k % num_layers
            # Extra entries go to the largest layers first to avoid over-sampling small ones.
            order = sorted(range(num_layers), key=lambda i: layer_sizes[i], reverse=True)
            per_layer = [base] * num_layers
            for i in range(remainder):
                per_layer[order[i]] += 1
            per_layer = [min(n, sz) for n, sz in zip(per_layer, layer_sizes)]
            indices = []
            offset = 0
            for n, sz in zip(per_layer, layer_sizes):
                if n > 0:
                    layer_vals = g_obs_det[offset:offset + sz]
                    if n >= sz:
                        local_idx = torch.arange(sz, device=g_obs_det.device)
                    else:
                        local_idx = torch.topk(layer_vals.abs(), k=n, largest=True).indices
                    indices.append(local_idx + offset)
                offset += sz
            selected_idx = (torch.cat(indices) if indices
                            else torch.arange(min(k, total_entries), device=g_obs_det.device))
        elif select_mode == "qr_pivot":
            # qr_pivot reorders rows after J_max is built; use topk_abs to seed the initial pool.
            selected_idx = torch.topk(g_obs_det.abs(), k=k, largest=True).indices
        else:
            raise ValueError(f"Unknown select_mode: {select_mode}")

    used_entries = selected_idx.numel()
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
                g_col = g_obs_d[selected_idx.to(g_obs_d.device)]
                col = fwAD.unpack_dual(g_col).tangent
                if col is None:
                    raise RuntimeError(
                        "Forward-mode AD did not propagate through autograd.grad. "
                        "Requires PyTorch >= 2.0. Upgrade PyTorch or open an issue."
                    )
                J[:, j] = col.detach().reshape(-1).to(device_for_J)
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

    return J, total_entries, unknowns


def _rank_of_J(J, print_svd_info=False):
    """Row-normalise J and return its numerical rank via matrix_rank."""
    # Normalize rows so σ_max(J_norm) ≤ √M instead of O(M).
    row_norms = torch.norm(J, dim=1, keepdim=True).clamp_min(1e-30)
    J_norm = J / row_norms
    if print_svd_info:
        svd_vals = torch.linalg.svdvals(J_norm)
        bottom = svd_vals[-10:].tolist()
        print(f"  [SVD] M={J.shape[0]}, bottom-10 singular values: "
              f"{[f'{v:.3e}' for v in bottom]}", flush=True)
    return int(torch.linalg.matrix_rank(J_norm).item())


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


def compute_jacobian_rank_sweep(
    net,
    x_norm,
    y,
    criterion,
    keep_ids=None,
    entry_masks=None,
    row_counts=None,
    select_mode="topk_abs",
    device_for_J="cpu",
    print_svd_info=False,
):
    """Rank of J for multiple row counts, building J once at max(row_counts).

    Returns {k: (rank, shape, used_entries, unknowns)} for each k in row_counts.
    Equivalent to calling compute_jacobian_rank separately for each k, but
    reuses the single forward+backward pass and J build.
    """
    if not row_counts:
        raise ValueError("row_counts must be a non-empty list")

    J_max, total_entries, unknowns = _build_jacobian(
        net, x_norm, y, criterion, keep_ids, entry_masks, max(row_counts), select_mode, device_for_J,
    )

    if select_mode == "qr_pivot":
        if not _SCIPY_AVAILABLE:
            raise ImportError("qr_pivot requires scipy. Install with: pip install scipy")
        # QR with column pivoting on J^T: pivots[i] is the i-th most linearly independent row of J.
        _, _, pivots = _scipy_linalg.qr(J_max.numpy().T, pivoting=True, mode="economic")
        J_max = J_max[torch.from_numpy(pivots.astype(np.int64))]

    results = {}
    for k in row_counts:
        J_k = J_max[:k]  # if k > J_max.shape[0], returns all rows
        rank = _rank_of_J(J_k, print_svd_info=print_svd_info)
        results[k] = (rank, tuple(J_k.shape), J_k.shape[0], unknowns)

    return results


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
