"""
Plot masking-sweep results from sweep CSV rows that reference masked registry entries.

Example:
    python helper/plot_masking_sweep_csv.py \
        results/masking_sweeps/mse_resnet18_cifar100_gradsize_topfrac_entries_layer_<hash>.csv \
        --out_dir results/masking_sweep_plots

Default threshold: --threshold_mse 0.01.
The summary CSV includes network/dataset columns, and plot titles show both.
Output filenames include network, dataset, and threshold, for example:
    sweep_plot_vgg13_cifar100_threshold_0p01.png

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
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from functions.io_utils import find_registry_entry, safe_makedirs, safe_savefig, safe_write

sns.set_theme(style='whitegrid')


def _default_registry_path(csv_path):
    results_dir = os.path.dirname(os.path.dirname(os.path.abspath(csv_path)))
    return os.path.join(results_dir, 'baselines', 'masked_registry_v2.json')


def _default_baseline_registry_path(csv_path):
    results_dir = os.path.dirname(os.path.dirname(os.path.abspath(csv_path)))
    return os.path.join(results_dir, 'baselines', 'idlg_baselines_registry_v2.json')


def _load_registry(path):
    with open(path) as f:
        return json.load(f)


def _load_registry_with_legacy(path):
    legacy_path = path.replace('_registry_v2.json', '_registry.json')
    registry = _load_registry(legacy_path) if os.path.isfile(legacy_path) else {}
    if os.path.isfile(path):
        registry.update(_load_registry(path))
    return registry


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
            entry_args = entry.get('args', {})
            topfrac = float(row['topfrac'])
            rows.append({
                **row,
                'args': entry_args,
                'network': entry_args.get('network', ''),
                'dataset': entry_args.get('dataset', ''),
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
            'tv_weight',
            'optimizer',
            'num_restarts',
            'max_iteration',
            'history_size',
        ]
        if key in masked_args
    }
    baseline_key = _baseline_key_from_masked_args(masked_args)
    stored_baseline_key, entry = find_registry_entry(baseline_registry, baseline_key, baseline_args)
    if entry is None:
        print(f'No matching baseline registry entry found for baseline_key={baseline_key}')
        return None
    baseline_key = stored_baseline_key
    mses = [float(v) for v in entry.get('best_mse_list', [])]
    if not mses:
        return None
    return {
        'command': '',
        'topfrac': 1.0,
        'masked_key': '',
        'baseline_key': baseline_key,
        'args': masked_args,
        'network': masked_args.get('network', ''),
        'dataset': masked_args.get('dataset', ''),
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


def _warn_if_mixed_sweep_rows(rows):
    sample_counts = sorted({row['n_total'] for row in rows})
    if len(sample_counts) > 1:
        print(
            'WARNING: Sweep rows contain different sample counts '
            f'{sample_counts}; reconstructed-image counts are not directly comparable.'
        )

    baseline_keys = {_baseline_key_from_masked_args(row.get('args', {})) for row in rows}
    if len(baseline_keys) > 1:
        print(
            'WARNING: Sweep rows map to different baseline configurations. '
            'The plotted baseline is derived from the first sweep row.'
        )


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
    return sorted(summary, key=lambda r: (r['pct_masked'], r['source'] != 'baseline'))


def _metadata(rows):
    first = rows[0] if rows else {}
    network = first.get('network') or first.get('args', {}).get('network') or 'unknown_network'
    dataset = first.get('dataset') or first.get('args', {}).get('dataset') or 'unknown_dataset'
    return network, dataset


def _safe_filename_part(value):
    return ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in str(value))


def _threshold_part(threshold):
    return str(threshold).replace('.', 'p').replace('-', 'm')


def _output_suffix(rows, threshold):
    network, dataset = _metadata(rows)
    return (
        f"{_safe_filename_part(network)}_"
        f"{_safe_filename_part(dataset)}_"
        f"threshold_{_threshold_part(threshold)}"
    )


def _write_summary(rows, threshold, out_dir):
    path = os.path.join(out_dir, f"masking_sweep_summary_{_output_suffix(rows, threshold)}.csv")
    fields = [
        'network',
        'dataset',
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


def _plot_title(rows):
    network, dataset = _metadata(rows)
    if network == 'unknown_network':
        network = ''
    if dataset == 'unknown_dataset':
        dataset = ''
    if network and dataset:
        return f'Masking sweep: {network} / {dataset}'
    if network or dataset:
        return f"Masking sweep: {network or dataset}"
    return 'Masking sweep'


def _rotate_xlabels(ax):
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')
        label.set_fontsize(8)


def _save_plot(fig, ax, path, title, ylabel, n_total, ymin, grid_alpha=None):
    ax.set_xlabel('Gradient entries masked (%)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(ymin, n_total + 0.5)
    _rotate_xlabels(ax)
    if grid_alpha is not None:
        ax.grid(True, alpha=grid_alpha)
    fig.tight_layout()
    if safe_savefig(fig, path, dpi=150):
        print(f'Saved: {path}')
    plt.close(fig)


def _plot(rows, threshold, out_dir):
    plot_labels = []
    for row in rows:
        pct_label = f"{row['pct_masked']:.0f}%"
        if row.get('source') == 'baseline':
            plot_labels.append(f'{pct_label} baseline')
        elif row['pct_masked'] == 0.0:
            plot_labels.append(f'{pct_label} masked')
        else:
            plot_labels.append(pct_label)

    n_recon = [r['n_reconstructed'] for r in rows]
    n_total = max(r['n_total'] for r in rows)
    title = _plot_title(rows)
    ylabel = f'Images reconstructed (MSE <= {threshold})'
    suffix = _output_suffix(rows, threshold)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(
        x=plot_labels,
        y=n_recon,
        marker='o',
        linewidth=1.8,
        ax=ax,
    )
    path = os.path.join(out_dir, f'sweep_plot_{suffix}.png')
    _save_plot(fig, ax, path, title, ylabel, n_total, ymin=-0.5, grid_alpha=0.3)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(
        x=plot_labels,
        y=n_recon,
        color='steelblue',
        edgecolor='black',
        linewidth=0.6,
        ax=ax,
    )
    path = os.path.join(out_dir, f'sweep_bar_{suffix}.png')
    _save_plot(fig, ax, path, title, ylabel, n_total, ymin=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path', help='CSV produced by normal iDLG_mask.py gradsize_topfrac_entries_layer runs')
    parser.add_argument('--threshold_mse', type=float, default=0.01)
    parser.add_argument('--registry_path', default=None,
                        help='Masked registry JSON. Defaults to merged legacy and v2 registries from the sweep CSV.')
    parser.add_argument('--baseline_registry_path', default=None,
                        help='iDLG baseline registry JSON. Defaults to merged legacy and v2 registries from the sweep CSV.')
    parser.add_argument('--no_baseline', action='store_true',
                        help='Do not include the matching iDLG baseline as the 0%% masked point.')
    parser.add_argument('--out_dir', default=None)
    args = parser.parse_args()

    registry_path = args.registry_path or _default_registry_path(args.csv_path)
    registry = _load_registry(registry_path) if args.registry_path else _load_registry_with_legacy(registry_path)
    rows = _read_rows(args.csv_path, registry)
    if not rows:
        print('No matching rows found.')
        return
    rows = _dedupe_latest(rows)
    _warn_if_mixed_sweep_rows(rows)
    if not args.no_baseline:
        baseline_registry_path = args.baseline_registry_path or _default_baseline_registry_path(args.csv_path)
        baseline_registry = (
            _load_registry(baseline_registry_path)
            if args.baseline_registry_path
            else _load_registry_with_legacy(baseline_registry_path)
        )
        baseline = _baseline_row(rows, baseline_registry)
        if baseline is not None:
            rows.append(baseline)
    rows = _summarise(rows, args.threshold_mse)

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.csv_path), 'masking_sweep_plots')
    if not safe_makedirs(out_dir):
        return

    _write_summary(rows, args.threshold_mse, out_dir)
    _plot(rows, args.threshold_mse, out_dir)


if __name__ == '__main__':
    main()
