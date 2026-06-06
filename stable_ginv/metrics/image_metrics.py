"""Image-quality metrics: PSNR from MSE, batched SSIM, and total variation."""
import math

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim


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
