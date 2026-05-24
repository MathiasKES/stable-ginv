"""
MSE calibration and masking sweep, parallelised across GPUs.

Calibration (--mse_visualise, no --threshold_mse):
  Run baseline iDLG on N images → sorted_mse.png, recon_grid.png, mse_results.csv

Sweep (--mse_visualise --threshold_mse X):
  Run args.mask_mode at topfrac 0.1→1.0 → sweep_plot.png, sweep_results.csv

Both imported and called from iDLG_mask.py.
"""
import os
import csv
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torchvision import transforms
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import functions.consts as consts
from helper.Network import get_model, weights_init
from helper.metrics import total_variation
from helper.training_utils import make_scheduler
from functions.masking import (build_gradient_mask, flatten_observed_gradients,
                                _get_last_fc_param_indices)


SWEEP_FRACS = [round(f * 0.1, 1) for f in range(1, 11)]   # 0.1 … 1.0


# ---- Per-worker setup -----------------------------------------------------------

def _setup(args, channel, num_classes, shape_img, device):
    """Build network, criterion, normalisation tensors, and pixel clamp bounds."""
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
    return net, criterion, dm, ds, lb, ub


# ---- Single experiment ----------------------------------------------------------

def _run_one(idx_net, dst, net, criterion, dm, ds, lb, ub, args, device,
             mask_mode=None, topfrac=None, return_images=False):
    """
    One iDLG experiment.
    mask_mode=None → baseline (no masking, full gradient vector).
    mask_mode set  → entry-wise mask applied at the given topfrac.
    return_images=True → include gt_np and recon_np in the returned dict.
    Returns best_mse (float) or a result dict when return_images=True.
    """
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

    named_params = list(net.named_parameters())
    last_fc_ids = _get_last_fc_param_indices(net)
    fw_idx = next(i for i in sorted(last_fc_ids) if named_params[i][0].endswith('.weight'))
    label_pred = torch.argmin(torch.sum(orig[fw_idx], dim=-1), dim=-1).detach().reshape((1,))

    if mask_mode is not None:
        _, entry_masks = build_gradient_mask('masked', mask_mode, net, orig,
                                             gradsize_topfrac=topfrac)
        gy = flatten_observed_gradients(orig, entry_masks=entry_masks)
    else:
        entry_masks = None
        gy = torch.cat([g.reshape(-1) for g in orig])

    def _cat_grads(grads):
        if entry_masks is not None:
            return flatten_observed_gradients(list(grads), entry_masks=entry_masks)
        return torch.cat([g.reshape(-1) for g in grads])

    def _grad_diff(gx):
        if args.grad_loss == 'cos':
            return 1.0 - F.cosine_similarity(gx.unsqueeze(0), gy.unsqueeze(0), dim=1, eps=1e-12)[0]
        return ((gx - gy) ** 2).sum()

    best_mse = float('inf')
    best_recon_np = None

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
                        net.parameters(), create_graph=True)
                    diff = _grad_diff(_cat_grads(dummy_grads))
                    if args.tv_weight > 0:
                        diff = diff + args.tv_weight * total_variation(dummy)
                    diff.backward()
                    return diff
                opt.step(closure)
                current_loss = closure().item()
                with torch.no_grad():
                    dummy.clamp_(lb, ub)
                if current_loss < 1e-6:
                    break
        else:
            signed = args.optimizer in ('signed_adam', 'signed_adamw')
            wd = 1e-5 if args.optimizer in ('adamw', 'signed_adamw') else 0.0
            cls = torch.optim.AdamW if args.optimizer in ('adamw', 'signed_adamw') else torch.optim.Adam
            opt = cls([dummy], lr=args.lr, weight_decay=wd)
            sched = make_scheduler(opt, args.iteration, args.gamma)
            for _ in range(args.iteration):
                opt.zero_grad()
                dummy_grads = torch.autograd.grad(
                    criterion(net(dummy), label_pred),
                    net.parameters(), create_graph=True)
                diff = _grad_diff(_cat_grads(dummy_grads))
                if args.tv_weight > 0:
                    diff = diff + args.tv_weight * total_variation(dummy)
                diff.backward()
                if signed and dummy.grad is not None:
                    with torch.no_grad():
                        dummy.grad.sign_()
                opt.step()
                sched.step()
                with torch.no_grad():
                    dummy.clamp_(lb, ub)
                if diff.item() < 1e-6:
                    break

        current_x = (dummy.detach() * ds + dm).clamp(0.0, 1.0)
        mse = torch.mean((current_x - gt_data) ** 2).item()
        if mse < best_mse:
            best_mse = mse
            if return_images:
                best_recon_np = current_x.cpu().numpy()[0]

    if return_images:
        return {'idx': int(img_idx), 'best_mse': best_mse,
                'gt_np': gt_data.cpu().numpy()[0], 'recon_np': best_recon_np}
    return best_mse


# ---- Parallel dispatcher --------------------------------------------------------

def _worker_fn(idx_net, device_idx, dst, args, channel, num_classes, shape_img,
               device, mask_mode, topfrac, return_images, result_queue):
    try:
        net, criterion, dm, ds, lb, ub = _setup(args, channel, num_classes, shape_img, device)
        result = _run_one(idx_net, dst, net, criterion, dm, ds, lb, ub,
                          args, device, mask_mode=mask_mode, topfrac=topfrac,
                          return_images=return_images)
        result_queue.put({'idx_net': idx_net, 'device_idx': device_idx, 'result': result})
    except Exception as e:
        result_queue.put({'idx_net': idx_net, 'device_idx': device_idx,
                          'result': None, 'error': str(e)})


def _init_mp():
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    mp.set_sharing_strategy('file_system')


def _run_parallel(n_exp, dst, args, channel, num_classes, shape_img,
                  mask_mode=None, topfrac=None, return_images=False, desc='experiments'):
    """
    Dispatch n_exp experiments across all available GPUs (or CPU).
    Returns list of results in experiment order.
    """

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    n_slots = max(num_gpus, 1)
    devices = [f'cuda:{i}' for i in range(num_gpus)] if num_gpus > 0 else ['cpu']

    result_queue = mp.SimpleQueue()
    jobs = list(range(n_exp))
    slot_busy = [False] * n_slots
    active_procs = {}  # slot -> Process
    results = [None] * n_exp
    n_done = 0

    with tqdm(total=n_exp, desc=f'  {desc}') as pbar:
        while n_done < n_exp:
            while jobs and not all(slot_busy):
                slot = next(i for i, busy in enumerate(slot_busy) if not busy)
                idx_net = jobs.pop(0)
                p = mp.Process(
                    target=_worker_fn,
                    args=(idx_net, slot, dst, args, channel, num_classes, shape_img,
                          devices[slot], mask_mode, topfrac, return_images, result_queue),
                )
                p.start()
                slot_busy[slot] = True
                active_procs[slot] = p

            item = result_queue.get()
            slot = item['device_idx']
            active_procs.pop(slot).join()
            slot_busy[slot] = False
            results[item['idx_net']] = item['result']
            if 'error' in item:
                print(f'\n  WARNING: exp {item["idx_net"]} failed: {item["error"]}')
            n_done += 1
            pbar.update(1)

    return results


# ---- Calibration output helpers -------------------------------------------------

def _to_display(img_np, channel):
    """[C,H,W] numpy → displayable (H,W) or (H,W,3) for imshow."""
    if channel == 1:
        return img_np[0]
    return np.clip(np.transpose(img_np, (1, 2, 0)), 0.0, 1.0)


def _save_sorted_mse(results, save_dir, threshold):
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


def _save_recon_grid(results, channel, save_dir, threshold):
    sorted_results = sorted(results, key=lambda r: r['best_mse'])
    n = len(sorted_results)
    cmap = 'gray' if channel == 1 else None
    fig, axes = plt.subplots(2, n, figsize=(max(8, n * 1.5), 4),
                              gridspec_kw={'hspace': 0.05, 'wspace': 0.05}, squeeze=False)
    for col, r in enumerate(sorted_results):
        mse = r['best_mse']
        reconstructed = threshold is not None and mse <= threshold
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
        title += f'   |   green <= {threshold} ({n_ok}/{n} reconstructed)'
    fig.suptitle(title, fontsize=8, y=1.01)
    fig.tight_layout()
    path = os.path.join(save_dir, 'recon_grid.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def _save_calibration_csv(results, save_dir, threshold):
    path = os.path.join(save_dir, 'mse_results.csv')
    fieldnames = ['rank', 'idx', 'best_mse']
    if threshold is not None:
        fieldnames.append('reconstructed')
    sorted_results = sorted(results, key=lambda r: r['best_mse'])
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rank, r in enumerate(sorted_results, 1):
            row = {'rank': rank, 'idx': r['idx'], 'best_mse': r['best_mse']}
            if threshold is not None:
                row['reconstructed'] = r['best_mse'] <= threshold
            w.writerow(row)
    print(f'Saved: {path}')


# ---- Public API -----------------------------------------------------------------

def run_mse_calibration(args, dst, channel, num_classes, shape_img, save_path):
    """
    Run baseline iDLG (no masking) on --num_exp images in parallel across GPUs.
    Called from iDLG_mask.main() when --mse_visualise is set without --threshold_mse.
    """
    _init_mp()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(save_path, f'threshold_{args.network}_{args.dataset}_{ts}')
    os.makedirs(out_dir, exist_ok=True)

    threshold = args.threshold_mse
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f'Calibration: {args.network} / {args.dataset}  {args.num_exp} images  '
          f'iter={args.iteration}  GPUs={max(num_gpus, 1)}'
          + (f'  threshold={threshold}' if threshold else ''))
    print(f'Saving to: {out_dir}\n')

    results = _run_parallel(args.num_exp, dst, args, channel, num_classes, shape_img,
                            mask_mode=None, return_images=True, desc='baseline iDLG')
    results = [r for r in results if r is not None]

    mses = [r['best_mse'] for r in results]
    print(f'\nMSE  min={min(mses):.6f}  max={max(mses):.6f}  '
          f'mean={np.mean(mses):.6f}  median={np.median(mses):.6f}')
    if threshold is not None:
        n_ok = sum(1 for m in mses if m <= threshold)
        print(f'Reconstructed (MSE <= {threshold}): {n_ok}/{len(mses)}')

    _save_sorted_mse(results, out_dir, threshold)
    _save_recon_grid(results, channel, out_dir, threshold)
    _save_calibration_csv(results, out_dir, threshold)

    if threshold is None:
        print('\nInspect sorted_mse.png and recon_grid.png, then re-run with --threshold_mse <value>.')


def run_mse_sweep(args, dst, channel, num_classes, shape_img, save_path):
    """
    Sweep args.mask_mode from topfrac 0.1 to 1.0 in parallel across GPUs.
    Called from iDLG_mask.main() when --mse_visualise and --threshold_mse are both set.
    """
    _init_mp()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(save_path, f'sweep_{args.network}_{args.dataset}_{ts}')
    os.makedirs(out_dir, exist_ok=True)

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f'MSE sweep: {args.mask_mode}  {len(SWEEP_FRACS)} fracs  '
          f'({len(SWEEP_FRACS) * args.num_exp} total runs)  '
          f'GPUs={max(num_gpus, 1)}  threshold={args.threshold_mse}')
    print(f'Saving to: {out_dir}\n')

    csv_rows = []
    for pt, frac in enumerate(SWEEP_FRACS, 1):
        print(f'[{pt}/{len(SWEEP_FRACS)}] topfrac={frac:.1f}  ({int((1 - frac) * 100)}% masked)')
        mses = _run_parallel(args.num_exp, dst, args, channel, num_classes, shape_img,
                             mask_mode=args.mask_mode, topfrac=frac,
                             desc=f'topfrac={frac:.1f}')
        mses = [m for m in mses if m is not None]
        n_recon = sum(1 for m in mses if m <= args.threshold_mse)
        row = {
            'mask_mode':       args.mask_mode,
            'topfrac':         frac,
            'pct_masked':      round((1.0 - frac) * 100.0, 1),
            'n_reconstructed': n_recon,
            'n_total':         len(mses),
            'avg_mse':         float(np.mean(mses)),
            'median_mse':      float(np.median(mses)),
        }
        csv_rows.append(row)
        print(f'  -> {n_recon}/{len(mses)} reconstructed  avg_mse={row["avg_mse"]:.6f}')

    _save_sweep_csv(csv_rows, out_dir)
    _save_plot(csv_rows, out_dir, args.threshold_mse, args.num_exp,
               args.network, args.dataset, args.mask_mode)


def _save_sweep_csv(rows, out_dir):
    path = os.path.join(out_dir, 'sweep_results.csv')
    fields = ['mask_mode', 'topfrac', 'pct_masked', 'n_reconstructed',
              'n_total', 'avg_mse', 'median_mse']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f'Saved: {path}')


def _save_plot(rows, out_dir, threshold, num_exp, network, dataset, mask_mode):
    rows = sorted(rows, key=lambda r: r['topfrac'])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot([r['topfrac'] for r in rows],
            [r['n_reconstructed'] for r in rows],
            marker='o', linewidth=1.8, label=mask_mode)
    ax.set_xlabel('Fraction of gradient entries shared (topfrac)')
    ax.set_ylabel(f'Images reconstructed  (MSE <= {threshold})')
    ax.set_title(f'Masking sweep — {network} / {dataset}  ({num_exp} images per point)\n'
                 f'{mask_mode}')
    ax.set_xlim(0.05, 1.05)
    ax.set_ylim(-0.5, num_exp + 0.5)
    ax.set_xticks(SWEEP_FRACS)
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
