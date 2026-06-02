"""
Plot paired baseline and masked reconstruction metrics from registry entries.

Example:
    python helper/plot_paired_masking_violin.py \
        --baseline_key 0c874fed9a0fd7cfd7467b380dead189 \
        --masked_key b9473c4448bd3cec04ed9bfa37c34561 \
        --out_dir results

By default, entries are read from:
    <resolved results path>/baselines/idlg_baselines_registry[_v2].json
    <resolved results path>/baselines/masked_registry[_v2].json
"""
import argparse
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from functions.io_utils import resolve_storage_paths, safe_makedirs, safe_savefig, safe_write


METRICS = [
    ("psnr", "PSNR (dB)"),
    ("mse", "MSE"),
    ("ssim", "SSIM"),
]


def _load_entry(registry, key, label):
    try:
        return registry[key]
    except KeyError as exc:
        raise ValueError(f"{label} key not found: {key}") from exc


def _load_registry(path):
    with open(path) as f:
        return json.load(f)


def _load_default_registry(baseline_dir, name):
    legacy_path = os.path.join(baseline_dir, f"{name}.json")
    v2_path = os.path.join(baseline_dir, f"{name}_v2.json")
    registry = {}
    found_paths = []
    for path in (legacy_path, v2_path):
        if os.path.isfile(path):
            registry.update(_load_registry(path))
            found_paths.append(path)
    if not found_paths:
        raise FileNotFoundError(
            f"No registry files found. Checked: {legacy_path}, {v2_path}"
        )
    print(f"Loaded: {', '.join(found_paths)}")
    return registry


def _paired_rows(baseline_entry, masked_entry):
    metric_values = {}
    expected_count = None
    for metric, _ in METRICS:
        baseline_values = baseline_entry.get(f"best_{metric}_list", [])
        masked_values = masked_entry.get(f"best_{metric}_list", [])
        if len(baseline_values) != len(masked_values):
            raise ValueError(
                f"{metric.upper()} sample counts differ: "
                f"baseline={len(baseline_values)}, masked={len(masked_values)}"
            )
        if expected_count is None:
            expected_count = len(baseline_values)
        elif len(baseline_values) != expected_count:
            raise ValueError("Metric lists do not all contain the same number of samples")
        metric_values[metric] = (baseline_values, masked_values)

    rows = []
    for sample_index in range(expected_count or 0):
        row = {
            "sample_index": sample_index,
            "sample_number": sample_index + 1,
        }
        for metric, _ in METRICS:
            baseline_value = float(metric_values[metric][0][sample_index])
            masked_value = float(metric_values[metric][1][sample_index])
            row[f"baseline_{metric}"] = baseline_value
            row[f"masked_{metric}"] = masked_value
            row[f"delta_{metric}"] = masked_value - baseline_value
        rows.append(row)
    return rows


def _long_dataframe(rows):
    records = []
    for row in rows:
        for metric, _ in METRICS:
            for method in ("Baseline", "Masked"):
                records.append({
                    "sample_number": row["sample_number"],
                    "metric": metric,
                    "method": method,
                    "value": row[f"{method.lower()}_{metric}"],
                })
    return pd.DataFrame(records)


def _write_rows(rows, path):
    fieldnames = list(rows[0]) if rows else [
        "sample_index",
        "sample_number",
        "baseline_psnr",
        "masked_psnr",
        "delta_psnr",
        "baseline_mse",
        "masked_mse",
        "delta_mse",
        "baseline_ssim",
        "masked_ssim",
        "delta_ssim",
    ]

    def _write(f):
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if safe_write(path, _write, newline=""):
        print(f"Saved: {path}")


def _plot(rows, path, title):
    dataframe = _long_dataframe(rows)
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, len(METRICS), figsize=(12, 4))
    palette = {"Baseline": "#4C72B0", "Masked": "#DD8452"}
    for ax, (metric, ylabel) in zip(axes, METRICS):
        metric_data = dataframe[dataframe["metric"] == metric]
        sns.violinplot(
            data=metric_data,
            x="method",
            y="value",
            hue="method",
            palette=palette,
            inner="quart",
            cut=0,
            legend=False,
            ax=ax,
        )
        sns.stripplot(
            data=metric_data,
            x="method",
            y="value",
            color="black",
            alpha=0.45,
            jitter=0.12,
            size=2.5,
            ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
    fig.suptitle(title)
    fig.tight_layout()
    if safe_savefig(fig, path, dpi=300, bbox_inches="tight"):
        print(f"Saved: {path}")
    plt.close(fig)


def _plot_deltas(rows, path, title):
    records = []
    for row in rows:
        records.extend([
            {"metric": "PSNR gain (dB)", "value": row["delta_psnr"]},
            {"metric": "MSE reduction", "value": -row["delta_mse"]},
            {"metric": "SSIM gain", "value": row["delta_ssim"]},
        ])
    dataframe = pd.DataFrame(records)
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric in zip(axes, dataframe["metric"].unique()):
        metric_data = dataframe[dataframe["metric"] == metric]
        sns.violinplot(
            data=metric_data,
            x="metric",
            y="value",
            color="#55A868",
            inner="quart",
            cut=0,
            ax=ax,
        )
        sns.stripplot(
            data=metric_data,
            x="metric",
            y="value",
            color="black",
            alpha=0.45,
            jitter=0.12,
            size=2.5,
            ax=ax,
        )
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("")
        ax.set_ylabel(metric)
        ax.set_xticklabels([])
    fig.suptitle(f"{title}\nPositive values mean improved reconstruction")
    fig.tight_layout()
    if safe_savefig(fig, path, dpi=300, bbox_inches="tight"):
        print(f"Saved: {path}")
    plt.close(fig)


def _print_extremes(rows):
    if not rows:
        print("No paired samples found.")
        return
    best = max(rows, key=lambda row: row["delta_psnr"])
    worst = min(rows, key=lambda row: row["delta_psnr"])
    print("Ranked by delta_psnr = masked_psnr - baseline_psnr")
    for label, row in [("Best", best), ("Worst", worst)]:
        print(
            f"{label}: sample_index={row['sample_index']}, "
            f"sample_number={row['sample_number']}, "
            f"baseline_psnr={row['baseline_psnr']:.6f}, "
            f"masked_psnr={row['masked_psnr']:.6f}, "
            f"delta_psnr={row['delta_psnr']:+.6f} dB"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Plot paired baseline and masked per-sample reconstruction metrics."
    )
    parser.add_argument(
        "registry_path",
        nargs="?",
        help="Optional JSON containing both entries. Overrides the default registry files.",
    )
    parser.add_argument(
        "--baseline_registry_path",
        default=None,
        help="Optional baseline registry JSON override. Ignored when registry_path is provided.",
    )
    parser.add_argument(
        "--masked_registry_path",
        default=None,
        help="Optional masked registry JSON override. Ignored when registry_path is provided.",
    )
    parser.add_argument("--baseline_key", required=True)
    parser.add_argument("--masked_key", required=True)
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--output_prefix", default="paired_masking")
    args = parser.parse_args()

    if args.registry_path:
        baseline_registry = masked_registry = _load_registry(args.registry_path)
    else:
        _, save_path = resolve_storage_paths(".")
        baseline_dir = os.path.join(save_path, "baselines")
        baseline_registry = (
            _load_registry(args.baseline_registry_path)
            if args.baseline_registry_path
            else _load_default_registry(baseline_dir, "idlg_baselines_registry")
        )
        masked_registry = (
            _load_registry(args.masked_registry_path)
            if args.masked_registry_path
            else _load_default_registry(baseline_dir, "masked_registry")
        )
    baseline_entry = _load_entry(baseline_registry, args.baseline_key, "Baseline")
    masked_entry = _load_entry(masked_registry, args.masked_key, "Masked")
    rows = _paired_rows(baseline_entry, masked_entry)

    if not safe_makedirs(args.out_dir):
        return
    csv_path = os.path.join(args.out_dir, f"{args.output_prefix}_per_sample.csv")
    plot_path = os.path.join(args.out_dir, f"{args.output_prefix}_violin.png")
    delta_plot_path = os.path.join(args.out_dir, f"{args.output_prefix}_delta_violin.png")
    _write_rows(rows, csv_path)
    baseline_args = baseline_entry.get("args", {})
    title = (
        f"Baseline vs masked reconstruction: "
        f"{baseline_args.get('network', 'unknown')} / "
        f"{baseline_args.get('dataset', 'unknown')}"
    )
    _plot(rows, plot_path, title)
    _plot_deltas(rows, delta_plot_path, title)
    _print_extremes(rows)


if __name__ == "__main__":
    main()
