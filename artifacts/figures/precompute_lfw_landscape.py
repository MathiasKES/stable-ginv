"""
Precompute the LFW gradient-matching loss landscape for
random_shooting_illustration.py.

Landscape model
---------------
Instead of a synthetic function or kernel regression, we use a soft-minimum
(log-sum-exp) over the REAL gradient-matching losses at each LFW image's
PCA position:

    L(x,y) = -T * log( Σ_i exp( -(loss_i + α·D_i²) / T ) )

where D_i² = ||(x,y) − pos_i||² and loss_i = 1 − cos(∇_W·, ∇_W_target).

This creates a basin of attraction centred on every real image.  The global
minimum is the target image (loss=0).  Other images with low gradient-matching
loss form local minima; images with high loss are absorbed by their neighbours.

Usage
-----
    python figures/precompute_lfw_landscape.py [--n_load 600] [--alpha 3.0] [--temp 0.012]
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, '..'))
from functions.Dataset import lfw_dataset
from helper.Network import LeNet, weights_init

from functions.io_utils import setstdout
setstdout()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_load",      type=int,   default=600)
    p.add_argument("--num_classes", type=int,   default=100)
    p.add_argument("--grid_res",    type=int,   default=300,
                   help="Side length of saved landscape grid.")
    p.add_argument("--alpha",       type=float, default=3.0,
                   help="Proximity weight: controls basin width vs. loss contrast.")
    p.add_argument("--temp",        type=float, default=0.012,
                   help="Log-sum-exp temperature: smaller = sharper basin boundaries.")
    p.add_argument("--seed",        type=int,   default=7)
    p.add_argument("--lfw_path",    type=str,
                   default=os.path.join(ROOT, '..', 'data', 'lfw'))
    p.add_argument("--out",         type=str,
                   default=os.path.join(ROOT, 'lfw_landscape.npz'))
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    SHAPE_IMG   = (32, 32)
    NUM_CLASSES = args.num_classes
    XLIM        = (-3.2, 2.8)
    YLIM        = (-2.2, 3.0)
    MARGIN      = 0.05
    ALPHA       = args.alpha
    TEMP        = args.temp

    # ── Load images ───────────────────────────────────────────────────────────
    dst = lfw_dataset(args.lfw_path, SHAPE_IMG)
    n   = min(args.n_load, len(dst))
    print(f"Loading {n} LFW grayscale images…")
    imgs_np, labels = [], []
    for i in range(n):
        img = Image.open(dst.imgs[i]).convert('L').resize(SHAPE_IMG)
        imgs_np.append(np.array(img, dtype=np.float32)[None] / 255.0)
        labels.append(int(dst.labs[i]))

    # ── Random LeNet ──────────────────────────────────────────────────────────
    model = LeNet(channel=1, num_classes=NUM_CLASSES, input_size=SHAPE_IMG)
    model.apply(weights_init)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    target_label = labels[0] % NUM_CLASSES

    def grad_vec(img_np):
        x = torch.tensor(img_np[None], dtype=torch.float32)
        y = torch.tensor([target_label])
        model.zero_grad()
        criterion(model(x), y).backward()
        return torch.cat(
            [p.grad.flatten() for p in model.parameters()]
        ).detach().numpy().copy()

    # ── Gradient-matching losses (all images vs. target, same label) ──────────
    print("Computing gradient-matching losses…")
    t0       = time.time()
    tgt_g    = grad_vec(imgs_np[0])
    tgt_norm = np.linalg.norm(tgt_g) + 1e-10

    all_losses = np.empty(n, dtype=np.float32)
    all_losses[0] = 0.0
    for i in range(1, n):
        g = grad_vec(imgs_np[i])
        all_losses[i] = float(1.0 - np.dot(g, tgt_g) / (np.linalg.norm(g) * tgt_norm + 1e-10))

    print(f"  {time.time()-t0:.1f}s  |  "
          f"loss ∈ [{all_losses.min():.5f}, {all_losses.max():.5f}]  "
          f"std={all_losses.std():.5f}")

    # ── PCA to 2D ─────────────────────────────────────────────────────────────
    print("Running PCA…")
    flat = np.stack([img.flatten() for img in imgs_np])
    mu   = flat.mean(0)
    _, _, Vt = np.linalg.svd(flat - mu, full_matrices=False)
    coords   = (flat - mu) @ Vt[:2].T   # (N, 2)

    def norm_coord(c, lo, hi):
        span = hi - lo
        return ((c - c.min()) / (c.max() - c.min())
                * span * (1 - 2*MARGIN) + lo + span*MARGIN).astype(np.float32)

    all_px = norm_coord(coords[:, 0], *XLIM)
    all_py = norm_coord(coords[:, 1], *YLIM)

    # ── Soft-minimum landscape on a grid ─────────────────────────────────────
    #   L(x,y) = -T · log( Σ_i exp( -(loss_i + α·D_i²) / T ) )
    #          = s_min − T · log( Σ_i exp( -(s_i − s_min) / T ) )
    #   where s_i = loss_i + α·D_i²
    #
    # The global minimum is the target image (loss_target = 0).
    # Other images with low gradient-matching loss form local minima.
    G  = args.grid_res
    xg = np.linspace(*XLIM, G).astype(np.float32)
    yg = np.linspace(*YLIM, G).astype(np.float32)
    XG, YG = np.meshgrid(xg, yg)
    Z  = np.empty((G, G), dtype=np.float32)

    print(f"Computing soft-min landscape on {G}×{G} grid (α={ALPHA}, T={TEMP})…")
    t0 = time.time()
    BATCH = 20
    for r0 in range(0, G, BATCH):
        r1  = min(r0 + BATCH, G)
        xb  = XG[r0:r1, :, None]              # (B, G, 1)
        yb  = YG[r0:r1, :, None]
        dx  = xb - all_px[None, None, :]      # (B, G, N)
        dy  = yb - all_py[None, None, :]
        raw = all_losses[None, None, :] + ALPHA * (dx*dx + dy*dy)   # (B, G, N)
        s_min  = raw.min(axis=2, keepdims=True)                      # (B, G, 1)
        lse    = np.log(np.exp(-(raw - s_min) / TEMP).sum(axis=2))  # (B, G)
        Z[r0:r1] = (s_min[:, :, 0] - TEMP * lse).astype(np.float32)

    print(f"  {time.time()-t0:.2f}s  |  Z ∈ [{Z.min():.5f}, {Z.max():.5f}]")

    # ── Save ──────────────────────────────────────────────────────────────────
    np.savez_compressed(
        args.out,
        all_px    = all_px,
        all_py    = all_py,
        all_losses= all_losses,
        target_px = np.float32(all_px[0]),
        target_py = np.float32(all_py[0]),
        Z_grid    = Z,
        xg_grid   = xg,
        yg_grid   = yg,
        xlim      = np.array(XLIM),
        ylim      = np.array(YLIM),
        alpha     = np.float32(ALPHA),
        temp      = np.float32(TEMP),
    )
    print(f"Saved: {args.out}")


if __name__ == '__main__':
    main()
