"""
Run baseline iDLG on N images and visualise the MSE distribution to help pick
a reconstruction-quality threshold for the masking-sweep graph.

Outputs saved to results/threshold_<TIMESTAMP>/:
  sorted_mse.png     MSEs sorted ascending — break points visible as jumps
  mse_histogram.png  Distribution of final best-MSE values
  recon_grid.png     GT + reconstruction for every experiment, sorted by MSE
  mse_results.csv    idx, best_mse, best_psnr [, reconstructed if --threshold given]

Usage (first pass — no threshold):
  python scripts/visualise_mse_threshold.py --network resnet18 --dataset cifar100 --num_exp 30

Usage (second pass — draw threshold line and label CSV):
  python scripts/visualise_mse_threshold.py ... --threshold 0.03
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import functions.consts as consts
from functions.Dataset import load_dataset
from helper.Network import get_model, weights_init
from helper.metrics import compute_psnr_from_mse, total_variation
from functions.masking import _get_last_fc_param_indices

from functions.io_utils import setstdout
setstdout()

def _run_one(idx_net, dst, net, dm, ds, lower_bound, upper_bound, criterion, args, device):
    """Run iDLG on a single experiment; return result dict."""
    seed = args.run_id + idx_net + 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tt = transforms.Compose([transforms.ToTensor()])
    idx_shuffle = np.random.permutation(len(dst))
    img_idx = idx_shuffle[0]

    gt_data = tt(dst[img_idx][0]).float().to(device).unsqueeze(0)
    gt_label = torch.tensor([dst[img_idx][1]], dtype=torch.long, device=device)
    gt_data_norm = (gt_data - dm) / ds

    out = net(gt_data_norm)
    loss = criterion(out, gt_label)
    dy_dx = torch.autograd.grad(loss, net.parameters())
    original_dy_dx = [g.detach().clone() for g in dy_dx]

    named_params = list(net.named_parameters())
    last_fc_ids = _get_last_fc_param_indices(net)
    final_weight_idx = next(
        i for i in sorted(last_fc_ids) if named_params[i][0].endswith('.weight')
    )
    label_pred = torch.argmin(
        torch.sum(original_dy_dx[final_weight_idx], dim=-1), dim=-1
    ).detach().reshape((1,))

    gy_cat = torch.cat([g.reshape(-1) for g in original_dy_dx]).detach()

    best_mse = float('inf')
    best_recon_np = None

    for _ in range(args.num_restarts):
        dummy_data = torch.randn_like(gt_data_norm).requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            [dummy_data], lr=args.lr,
            max_iter=args.max_iteration,
            history_size=args.history_size,
        )

        for _ in range(args.iteration):
            def closure():
                optimizer.zero_grad()
                pred = net(dummy_data)
                dummy_loss = criterion(pred, label_pred)
                dummy_dy_dx = torch.autograd.grad(
                    dummy_loss, net.parameters(), create_graph=True
                )
                gx_cat = torch.cat([g.reshape(-1) for g in dummy_dy_dx])
                grad_diff = 1.0 - F.cosine_similarity(
                    gx_cat.unsqueeze(0), gy_cat.unsqueeze(0), dim=1, eps=1e-12
                )[0]
                if args.tv_weight > 0:
                    grad_diff = grad_diff + args.tv_weight * total_variation(dummy_data)
                grad_diff.backward()
                return grad_diff

            optimizer.step(closure)
            closure()
            with torch.no_grad():
                dummy_data.clamp_(lower_bound, upper_bound)

        current_x = (dummy_data.detach() * ds + dm).clamp(0.0, 1.0)
        mse = torch.mean((current_x - gt_data) ** 2).item()
        if mse < best_mse:
            best_mse = mse
            best_recon_np = current_x.cpu().numpy()[0]

    return {
        'idx': int(img_idx),
        'best_mse': best_mse,
        'best_psnr': compute_psnr_from_mse(best_mse),
        'gt_np': gt_data.cpu().numpy()[0],
        'recon_np': best_recon_np,
    }


def _to_display(img_np, channel):
    """[C,H,W] numpy → displayable (H,W) or (H,W,3)."""
    if channel == 1:
        return img_np[0]
    return np.clip(np.transpose(img_np, (1, 2, 0)), 0.0, 1.0)


def save_sorted_mse(results, save_dir, threshold):
    mses = [r['best_mse'] for r in results]
    sorted_mses = sorted(mses)
    n = len(sorted_mses)

    fig, ax = plt.subplots(figsize=(max(6, n * 0.3 + 2), 4))
    ax.plot(range(1, n + 1), sorted_mses, marker='o', markersize=4, linewidth=1.5)
    if threshold is not None:
        ax.axhline(threshold, color='red', linestyle='--', linewidth=1.5,
                   label=f'threshold = {threshold}')
        n_below = sum(1 for m in mses if m <= threshold)
        ax.set_title(f'MSE sorted ascending  —  {n_below}/{n} images below threshold')
        ax.legend()
    else:
        ax.set_title('MSE sorted ascending  —  look for natural break points')
    ax.set_xlabel('Rank (best → worst)')
    ax.set_ylabel('Best MSE')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, 'sorted_mse.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'Saved: {path}')


def save_histogram(results, save_dir, threshold):
    mses = [r['best_mse'] for r in results]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(mses, bins=min(20, len(mses)), edgecolor='black', alpha=0.75)
    if threshold is not None:
        ax.axvline(threshold, color='red', linestyle='--', linewidth=1.5,
                   label=f'threshold = {threshold}')
        ax.legend()
    ax.set_xlabel('Best MSE')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of best MSE values')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, 'mse_histogram.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'Saved: {path}')


def save_recon_grid(results, channel, save_dir, threshold):
    sorted_results = sorted(results, key=lambda r: r['best_mse'])
    n = len(sorted_results)
    cmap = 'gray' if channel == 1 else None

    fig, axes = plt.subplots(
        2, n,
        figsize=(max(8, n * 1.5), 4),
        gridspec_kw={'hspace': 0.05, 'wspace': 0.05},
        squeeze=False,
    )

    for col, r in enumerate(sorted_results):
        mse = r['best_mse']
        reconstructed = (threshold is not None and mse <= threshold)
        color = 'green' if reconstructed else ('red' if threshold is not None else 'black')

        ax_gt = axes[0, col]
        ax_gt.imshow(_to_display(r['gt_np'], channel), cmap=cmap, interpolation='nearest')
        ax_gt.axis('off')
        if col == 0:
            ax_gt.set_ylabel('GT', fontsize=7)

        ax_r = axes[1, col]
        ax_r.imshow(_to_display(r['recon_np'], channel), cmap=cmap, interpolation='nearest')
        ax_r.axis('off')
        ax_r.set_xlabel(f'{mse:.4f}', fontsize=6, color=color, labelpad=1)
        if col == 0:
            ax_r.set_ylabel('Recon', fontsize=7)

    title = 'Reconstructions sorted by MSE (best → worst)'
    if threshold is not None:
        n_ok = sum(1 for r in results if r['best_mse'] <= threshold)
        title += f'   |   green ≤ {threshold} ({n_ok}/{n} reconstructed)'
    fig.suptitle(title, fontsize=8, y=1.01)
    fig.tight_layout()
    path = os.path.join(save_dir, 'recon_grid.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def save_csv(results, save_dir, threshold):
    path = os.path.join(save_dir, 'mse_results.csv')
    fieldnames = ['rank', 'idx', 'best_mse', 'best_psnr']
    if threshold is not None:
        fieldnames.append('reconstructed')

    sorted_results = sorted(results, key=lambda r: r['best_mse'])
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, r in enumerate(sorted_results, 1):
            row = {
                'rank': rank,
                'idx': r['idx'],
                'best_mse': r['best_mse'],
                'best_psnr': r['best_psnr'],
            }
            if threshold is not None:
                row['reconstructed'] = r['best_mse'] <= threshold
            writer.writerow(row)
    print(f'Saved: {path}')


def main():
    parser = argparse.ArgumentParser(
        description='Visualise iDLG MSE distribution to calibrate a reconstruction threshold.'
    )
    parser.add_argument('--network',      type=str,   default='resnet18')
    parser.add_argument('--dataset',      type=str,   default='cifar100')
    parser.add_argument('--pretrained',   action='store_true')
    parser.add_argument('--num_exp',      type=int,   default=20)
    parser.add_argument('--run_id',       type=int,   default=0)
    parser.add_argument('--iteration',    type=int,   default=300)
    parser.add_argument('--lr',           type=float, default=1.0)
    parser.add_argument('--max_iteration',type=int,   default=20)
    parser.add_argument('--history_size', type=int,   default=100)
    parser.add_argument('--num_restarts', type=int,   default=1)
    parser.add_argument('--tv_weight',    type=float, default=0.0)
    parser.add_argument('--device',       type=str,   default='cuda:0')
    parser.add_argument('--threshold',    type=float, default=None,
                        help='Draw this value as a line on the plots and label CSV.')
    args = parser.parse_args()

    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'
        print('CUDA not available, falling back to CPU.')

    if os.access('/work3/s234843/bachelor', os.R_OK | os.W_OK | os.X_OK):
        data_path = '/work3/s234843/bachelor/datasets'
        base_save = '/work3/s234843/bachelor/results'
    else:
        data_path = './data'
        base_save = './results'

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(base_save, f'threshold_{args.network}_{args.dataset}_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)

    dst, channel, num_classes, shape_img = load_dataset(args.dataset, data_path)

    net = get_model(args.network, channel=channel, num_classes=num_classes,
                    input_size=shape_img, pretrained=args.pretrained)
    if not args.pretrained and args.network in ('LeNet', 'LeNet_bigger', 'MediumCNN', 'BiggerCNN'):
        net.apply(weights_init)
    net = net.to(device).eval()

    criterion = nn.CrossEntropyLoss().to(device)

    if args.pretrained and channel == 3:
        dm = torch.tensor(consts.imagenet_mean, device=device).view(1, channel, 1, 1)
        ds = torch.tensor(consts.imagenet_std,  device=device).view(1, channel, 1, 1)
    else:
        dm = torch.tensor(getattr(consts, f'{args.dataset.lower()}_mean'), device=device).view(1, channel, 1, 1)
        ds = torch.tensor(getattr(consts, f'{args.dataset.lower()}_std'),  device=device).view(1, channel, 1, 1)

    lower_bound = (-dm / ds).to(device)
    upper_bound = ((1.0 - dm) / ds).to(device)

    print(f'Network: {args.network}  Dataset: {args.dataset}  '
          f'Experiments: {args.num_exp}  Iterations: {args.iteration}')
    if args.threshold is not None:
        print(f'Threshold: {args.threshold}')
    print(f'Saving to: {save_dir}\n')

    results = []
    for idx_net in tqdm(range(args.num_exp), desc='iDLG experiments'):
        r = _run_one(idx_net, dst, net, dm, ds, lower_bound, upper_bound,
                     criterion, args, device)
        results.append(r)
        tqdm.write(f'  exp {idx_net:3d}  idx={r["idx"]:5d}  '
                   f'mse={r["best_mse"]:.6f}  psnr={r["best_psnr"]:.2f} dB')

    mses = [r['best_mse'] for r in results]
    print(f'\nMSE  min={min(mses):.6f}  max={max(mses):.6f}  '
          f'mean={np.mean(mses):.6f}  median={np.median(mses):.6f}')
    if args.threshold is not None:
        n_ok = sum(1 for m in mses if m <= args.threshold)
        print(f'Reconstructed (MSE ≤ {args.threshold}): {n_ok}/{len(mses)}')

    save_sorted_mse(results, save_dir, args.threshold)
    save_histogram(results, save_dir, args.threshold)
    save_recon_grid(results, channel, save_dir, args.threshold)
    save_csv(results, save_dir, args.threshold)


if __name__ == '__main__':
    main()
