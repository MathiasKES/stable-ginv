"""
Plot masking-sweep results from sweep CSV rows that reference masked registry entries.

Example:
    python scripts/plot_masking_sweep_csv.py \
        results/masking_sweeps/mse_resnet18_cifar100_gradsize_topfrac_entries_layer_<hash>.csv \
        --threshold_mse 0.03 \
        --out_dir results/masking_sweep_plots

The input CSV is produced by:
    python iDLG_mask.py --methods masked \
        --mask_mode gradsize_topfrac_entries_layer --gradsize_topfrac 0.5
"""
import argparse
import csv
import hashlib
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from functions.io_utils import safe_makedirs, safe_savefig, safe_write


def _default_registry_path(csv_path):
    results_dir = os.path.dirname(os.path.dirname(os.path.abspath(csv_path)))
    return os.path.join(results_dir, 'baselines', 'masked_registry.json')


def _default_baseline_registry_path(csv_path):
    results_dir = os.path.dirname(os.path.dirname(os.path.abspath(csv_path)))
    return os.path.join(results_dir, 'baselines', 'idlg_baselines_registry.json')


def _load_registry(path):
    with open(path) as f:
        return json.load(f)


def _baseline_key_from_masked_args(masked_args):
    baseline_args = {
        key: masked_args[key]
        for key in [
            'dataset',
            'network',
            'pretrained',
            'lr',
            'gamma',
            'grad_loss',
            'num_dummy',
            'iteration',
            'num_exp',
            'run_id',
            'tv_weight',
            'optimizer',
            'num_restarts',
            'max_iteration',
            'history_size',
        ]
        if key in masked_args
    }
    key_json = json.dumps(baseline_args, sort_keys=True)
    return hashlib.md5(key_json.encode('utf-8')).hexdigest()


def _read_rows(path, registry):
    rows = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            masked_key = row.get('masked_key', '')
            entry = registry.get(masked_key)
            if entry is None:
                print(f"Skipping row with missing masked_key={masked_key}")
                continue
            mses = [float(v) for v in entry.get('best_mse_list', [])]
            if not mses:
                continue
            topfrac = float(row['topfrac'])
            rows.append({
                **row,
                'args': entry.get('args', {}),
                'topfrac': topfrac,
                'pct_masked': (1.0 - topfrac) * 100.0,
                'n_total': len(mses),
                'per_sample_mse': mses,
                'source': 'masked',
            })
    return rows


def _baseline_row(rows, baseline_registry):
    if not rows:
        return None
    masked_args = rows[0].get('args', {})
    baseline_key = _baseline_key_from_masked_args(masked_args)
    entry = baseline_registry.get(baseline_key)
    if entry is None:
        print(f'No matching baseline registry entry found for baseline_key={baseline_key}')
        return None
    mses = [float(v) for v in entry.get('best_mse_list', [])]
    if not mses:
        return None
    return {
        'command': '',
        'topfrac': 1.0,
        'masked_key': '',
        'baseline_key': baseline_key,
        'pct_masked': 0.0,
        'n_total': len(mses),
        'per_sample_mse': mses,
        'source': 'baseline',
    }


def _dedupe_latest(rows):
    latest = {}
    for row in rows:
        latest[row['topfrac']] = row
    return list(latest.values())


def _summarise(rows, threshold):
    summary = []
    for row in rows:
        mses = row['per_sample_mse']
        n_reconstructed = sum(1 for mse in mses if mse <= threshold)
        summary.append({
            **row,
            'n_reconstructed': n_reconstructed,
            'avg_mse': float(np.mean(mses)),
            'median_mse': float(np.median(mses)),
        })
    return sorted(summary, key=lambda r: r['pct_masked'])


def _write_summary(rows, out_dir):
    path = os.path.join(out_dir, 'masking_sweep_summary.csv')
    fields = [
        'source',
        'topfrac',
        'pct_masked',
        'n_reconstructed',
        'n_total',
        'avg_mse',
        'median_mse',
        'command',
    ]

    def _write(f):
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    if safe_write(path, _write, newline=''):
        print(f'Saved: {path}')


def _plot(rows, threshold, out_dir):
    pct = [r['pct_masked'] for r in rows]
    n_recon = [r['n_reconstructed'] for r in rows]
    n_total = max(r['n_total'] for r in rows)
    title = 'Masking sweep'
    ylabel = f'Images reconstructed (MSE <= {threshold})'

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(pct, n_recon, marker='o', linewidth=1.8)
    ax.set_xlabel('Gradient entries masked (%)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(-2.5, max(pct) + 2.5)
    ax.set_ylim(-0.5, n_total + 0.5)
    ax.set_xticks(pct)
    ax.set_xticklabels([f'{p:.0f}%' for p in pct], rotation=45, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, 'sweep_plot.png')
    if safe_savefig(fig, path, dpi=150):
        print(f'Saved: {path}')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(pct)), n_recon, color='steelblue', edgecolor='black', linewidth=0.6)
    ax.set_xlabel('Gradient entries masked (%)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(range(len(pct)))
    ax.set_xticklabels([f'{p:.0f}%' for p in pct], rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, n_total + 0.5)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, 'sweep_bar.png')
    if safe_savefig(fig, path, dpi=150):
        print(f'Saved: {path}')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path', help='CSV produced by normal iDLG_mask.py gradsize_topfrac_entries_layer runs')
    parser.add_argument('--threshold_mse', type=float, required=True)
    parser.add_argument('--registry_path', default=None,
                        help='Masked registry JSON. Defaults to ../baselines/masked_registry.json from the sweep CSV.')
    parser.add_argument('--baseline_registry_path', default=None,
                        help='iDLG baseline registry JSON. Defaults to ../baselines/idlg_baselines_registry.json from the sweep CSV.')
    parser.add_argument('--no_baseline', action='store_true',
                        help='Do not include the matching iDLG baseline as the 0% masked point.')
    parser.add_argument('--out_dir', default=None)
    args = parser.parse_args()

    registry_path = args.registry_path or _default_registry_path(args.csv_path)
    registry = _load_registry(registry_path)
    rows = _read_rows(args.csv_path, registry)
    if not rows:
        print('No matching rows found.')
        return
    rows = _dedupe_latest(rows)
    if not args.no_baseline:
        baseline_registry_path = args.baseline_registry_path or _default_baseline_registry_path(args.csv_path)
        baseline_registry = _load_registry(baseline_registry_path)
        baseline = _baseline_row(rows, baseline_registry)
        if baseline is not None:
            rows = [r for r in rows if r['pct_masked'] != 0.0]
            rows.append(baseline)
    rows = _summarise(rows, args.threshold_mse)

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.csv_path), 'masking_sweep_plots')
    if not safe_makedirs(out_dir):
        return

    _write_summary(rows, out_dir)
    _plot(rows, args.threshold_mse, out_dir)


if __name__ == '__main__':
    main()
