import math

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim

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
):
    """Rank of J = d vec(g_obs) / d vec(x), built row-by-row to avoid OOM."""
    params = tuple(net.parameters())
    x_norm = x_norm.detach().clone().requires_grad_(True)

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

    # Normalize rows so σ_max(J_norm) ≤ √M instead of O(M).
    # Without this, the standard threshold max(M,N)·ε·σ_max grows as O(M²·ε),
    # causing rank to decline as more rows are added.
    row_norms = torch.norm(J, dim=1, keepdim=True).clamp_min(1e-30)
    J_norm = J / row_norms

    # Floating-point-aware absolute threshold on the normalised matrix.
    # After normalization σ_max ≤ √M, so the precision floor grows as O(M^1.5·ε)
    # rather than O(M²·ε).  Take the larger of the user's rank_tol and this floor
    # so that neither numerical noise nor genuinely tiny singular values are counted.
    M, N = J_norm.shape
    eps = torch.finfo(J_norm.dtype).eps
    precision_atol = max(M, N) * eps * (M ** 0.5)
    atol = max(rank_tol, precision_atol)

    jac_rank = int(torch.linalg.matrix_rank(J_norm, atol=atol, rtol=0.0).item())

    return jac_rank, tuple(J.shape), used_entries, unknowns


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
