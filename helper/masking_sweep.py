"""
Entry-wise masking sweep: run iDLG at gradsize_topfrac_entries and
gradsize_topfrac_entries_layer for fractions 0.1, 0.2, ..., 1.0 and plot the
number of images reconstructed (best MSE <= threshold) vs. fraction shared.

Imported and called from iDLG_mask.py when --mse_visualise is set.
"""
import os
import csv
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import functions.consts as consts
from helper.Network import get_model, weights_init
from helper.metrics import total_variation
from functions.masking import (build_gradient_mask, flatten_observed_gradients,
                                _get_last_fc_param_indices)

SWEEP_FRACS = [round(f * 0.1, 1) for f in range(1, 11)]   # 0.1 … 1.0
SWEEP_MODES = ['gradsize_topfrac_entries', 'gradsize_topfrac_entries_layer']
_MODE_LABELS = {
    'gradsize_topfrac_entries':       'global (topfrac_entries)',
    'gradsize_topfrac_entries_layer': 'per-layer (topfrac_entries_layer)',
}
_MODE_COLORS = {
    'gradsize_topfrac_entries':       'steelblue',
    'gradsize_topfrac_entries_layer': 'darkorange',
}


def _run_one(idx_net, dst, net, criterion, dm, ds, lb, ub, args, device,
             mask_mode, topfrac):
    """One iDLG experiment with a fixed entry-wise mask; returns best MSE."""
    seed = args.run_id + idx_net + 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tt = transforms.Compose([transforms.ToTensor()])
    img_idx = np.random.permutation(len(dst))[0]
    gt_data = tt(dst[img_idx][0]).float().to(device).unsqueeze(0)
    gt_label = torch.tensor([dst[img_idx][1]], dtype=torch.long, device=device)
    gt_norm = (gt_data - dm) / ds

    out = net(gt_norm)
    loss = criterion(out, gt_label)
    dy_dx = torch.autograd.grad(loss, net.parameters())
    orig = [g.detach().clone() for g in dy_dx]

    _, entry_masks = build_gradient_mask('masked', mask_mode, net, orig,
                                         gradsize_topfrac=topfrac)
    gy = flatten_observed_gradients(orig, entry_masks=entry_masks)

    named_params = list(net.named_parameters())
    last_fc_ids = _get_last_fc_param_indices(net)
    fw_idx = next(i for i in sorted(last_fc_ids) if named_params[i][0].endswith('.weight'))
    label_pred = torch.argmin(torch.sum(orig[fw_idx], dim=-1), dim=-1).detach().reshape((1,))

    best_mse = float('inf')
    for _ in range(args.num_restarts):
        dummy = torch.randn_like(gt_norm).requires_grad_(True)

        if args.optimizer == 'lbfgs':
            opt = torch.optim.LBFGS([dummy], lr=args.lr,
                                     max_iter=args.max_iteration,
                                     history_size=args.history_size)
            for _ in range(args.iteration):
                def closure():
                    opt.zero_grad()
                    dummy_grads = torch.autograd.grad(
                        criterion(net(dummy), label_pred),
                        net.parameters(), create_graph=True,
                    )
                    gx = flatten_observed_gradients(list(dummy_grads), entry_masks=entry_masks)
                    if args.grad_loss == 'cos':
                        diff = 1.0 - F.cosine_similarity(
                            gx.unsqueeze(0), gy.unsqueeze(0), dim=1, eps=1e-12)[0]
                    else:
                        diff = ((gx - gy) ** 2).sum()
                    if args.tv_weight > 0:
                        diff = diff + args.tv_weight * total_variation(dummy)
                    diff.backward()
                    return diff
                opt.step(closure)
                closure()
                with torch.no_grad():
                    dummy.clamp_(lb, ub)
        else:
            cls = torch.optim.AdamW if args.optimizer == 'adamw' else torch.optim.Adam
            wd = 1e-5 if args.optimizer == 'adamw' else 0.0
            opt = cls([dummy], lr=args.lr, weight_decay=wd)
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=300, gamma=args.gamma)
            for _ in range(args.iteration):
                opt.zero_grad()
                dummy_grads = torch.autograd.grad(
                    criterion(net(dummy), label_pred),
                    net.parameters(), create_graph=True,
                )
                gx = flatten_observed_gradients(list(dummy_grads), entry_masks=entry_masks)
                if args.grad_loss == 'cos':
                    diff = 1.0 - F.cosine_similarity(
                        gx.unsqueeze(0), gy.unsqueeze(0), dim=1, eps=1e-12)[0]
                else:
                    diff = ((gx - gy) ** 2).sum()
                if args.tv_weight > 0:
                    diff = diff + args.tv_weight * total_variation(dummy)
                diff.backward()
                opt.step()
                sched.step()
                with torch.no_grad():
                    dummy.clamp_(lb, ub)

        mse = torch.mean(((dummy.detach() * ds + dm).clamp(0.0, 1.0) - gt_data) ** 2).item()
        if mse < best_mse:
            best_mse = mse

    return best_mse


def run_mse_sweep(args, dst, channel, num_classes, shape_img, save_path):
    """
    Sweep gradsize_topfrac_entries and gradsize_topfrac_entries_layer from 0.1 to 1.0.
    Called from iDLG_mask.main() when --mse_visualise is set.
    Requires args.threshold_mse to be set.
    """
    if args.threshold_mse is None:
        raise ValueError('--threshold_mse is required when --mse_visualise is set.')
    if args.optimizer in ('signed_adam', 'signed_adamw'):
        raise ValueError('signed_adam/signed_adamw are not supported in --mse_visualise mode.')

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
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
        dm = torch.tensor(getattr(consts, f'{args.dataset.lower()}_mean'),
                          device=device).view(1, channel, 1, 1)
        ds = torch.tensor(getattr(consts, f'{args.dataset.lower()}_std'),
                          device=device).view(1, channel, 1, 1)
    lb = (-dm / ds).to(device)
    ub = ((1.0 - dm) / ds).to(device)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(save_path, f'sweep_{args.network}_{args.dataset}_{ts}')
    os.makedirs(out_dir, exist_ok=True)

    n_pts = len(SWEEP_MODES) * len(SWEEP_FRACS)
    print(f'MSE sweep: {len(SWEEP_MODES)} modes x {len(SWEEP_FRACS)} fracs = {n_pts} points  '
          f'({n_pts * args.num_exp} total runs)  threshold={args.threshold_mse}')
    print(f'Saving to: {out_dir}\n')

    csv_rows = []
    pt = 0
    for mask_mode in SWEEP_MODES:
        for frac in SWEEP_FRACS:
            pt += 1
            print(f'[{pt}/{n_pts}] {mask_mode}  topfrac={frac:.1f}  '
                  f'({int((1 - frac) * 100)}% masked)')
            mses = []
            for idx_net in tqdm(range(args.num_exp), desc='  experiments', leave=False):
                mse = _run_one(idx_net, dst, net, criterion, dm, ds, lb, ub,
                               args, device, mask_mode, frac)
                mses.append(mse)
            n_recon = sum(1 for m in mses if m <= args.threshold_mse)
            row = {
                'mask_mode':       mask_mode,
                'topfrac':         frac,
                'pct_masked':      round((1.0 - frac) * 100.0, 1),
                'n_reconstructed': n_recon,
                'n_total':         len(mses),
                'avg_mse':         float(np.mean(mses)),
                'median_mse':      float(np.median(mses)),
            }
            csv_rows.append(row)
            print(f'  -> {n_recon}/{args.num_exp} reconstructed  '
                  f'avg_mse={row["avg_mse"]:.6f}')

    _save_csv(csv_rows, out_dir)
    _save_plot(csv_rows, out_dir, args.threshold_mse, args.num_exp,
               args.network, args.dataset)


def _save_csv(rows, out_dir):
    path = os.path.join(out_dir, 'sweep_results.csv')
    fields = ['mask_mode', 'topfrac', 'pct_masked', 'n_reconstructed',
              'n_total', 'avg_mse', 'median_mse']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f'Saved: {path}')


def _save_plot(rows, out_dir, threshold, num_exp, network, dataset):
    fig, ax = plt.subplots(figsize=(9, 5))
    by_mode = {}
    for row in rows:
        by_mode.setdefault(row['mask_mode'], []).append(row)
    for mode, mode_rows in by_mode.items():
        mode_rows = sorted(mode_rows, key=lambda r: r['topfrac'])
        ax.plot([r['topfrac'] for r in mode_rows],
                [r['n_reconstructed'] for r in mode_rows],
                marker='o', linewidth=1.8,
                label=_MODE_LABELS.get(mode, mode),
                color=_MODE_COLORS.get(mode))
    ax.set_xlabel('Fraction of gradient entries shared (topfrac)')
    ax.set_ylabel(f'Images reconstructed  (MSE <= {threshold})')
    ax.set_title(f'Entry-wise masking sweep — {network} / {dataset}  '
                 f'({num_exp} images per point)')
    ax.set_xlim(0.05, 1.05)
    ax.set_ylim(-0.5, num_exp + 0.5)
    ax.set_xticks(SWEEP_FRACS)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(SWEEP_FRACS)
    ax2.set_xticklabels([f'{int((1 - f) * 100)}%' for f in SWEEP_FRACS])
    ax2.set_xlabel('Gradient entries masked (%)')
    fig.tight_layout()
    path = os.path.join(out_dir, 'sweep_plot.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'Saved: {path}')
