"""
Plot paired baseline and masked reconstruction metrics from registry entries.

Example:
    python helper/plot_paired_masking_violin.py \
        --baseline_key 0c874fed9a0fd7cfd7467b380dead189 \
        --masked_key b9473c4448bd3cec04ed9bfa37c34561 \
        --out_dir results

For a corrected historical CSV:
    python helper/plot_paired_masking_violin.py \
        --paired_csv_path results/50_per_sample_corrected.csv \
        --out_dir . \
        --output_prefix corrected_50

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
METRIC_BY_NAME = dict(METRICS)
DELTA_LABELS = {
    "psnr": ("PSNR gain (dB)", lambda row: row["delta_psnr"]),
    "mse": ("MSE reduction", lambda row: -row["delta_mse"]),
    "ssim": ("SSIM gain", lambda row: row["delta_ssim"]),
}


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


def _load_paired_rows(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No paired sample rows found in: {path}")
    int_fields = {"run_id", "sample_index", "sample_number", "baseline_index", "masked_index"}
    metric_fields = {
        f"{prefix}_{metric}"
        for prefix in ("baseline", "masked", "delta")
        for metric, _ in METRICS
    }
    for row in rows:
        for field in int_fields:
            if field in row:
                row[field] = int(row[field])
        for field in metric_fields:
            row[field] = float(row[field])
    return rows


def _paired_rows(baseline_entry, masked_entry):
    metric_values = {}
    baseline_count = None
    masked_count = None
    for metric, _ in METRICS:
        baseline_values = baseline_entry.get(f"best_{metric}_list", [])
        masked_values = masked_entry.get(f"best_{metric}_list", [])
        if baseline_count is None:
            baseline_count = len(baseline_values)
            masked_count = len(masked_values)
        elif len(baseline_values) != baseline_count or len(masked_values) != masked_count:
            raise ValueError(
                "PSNR, MSE, and SSIM lists must have consistent lengths "
                "within each registry entry"
            )
        metric_values[metric] = (baseline_values, masked_values)

    baseline_start = int(baseline_entry.get("args", {}).get("run_id", 0))
    masked_start = int(masked_entry.get("args", {}).get("run_id", 0))
    overlap_start = max(baseline_start, masked_start)
    overlap_end = min(baseline_start + baseline_count, masked_start + masked_count)
    if overlap_start >= overlap_end:
        raise ValueError(
            "Baseline and masked registry entries do not cover any shared run_ids: "
            f"baseline={baseline_start}..{baseline_start + baseline_count - 1}, "
            f"masked={masked_start}..{masked_start + masked_count - 1}"
        )
    if baseline_count != masked_count or baseline_start != masked_start:
        print(
            "WARNING: Registry ranges differ; plotting only overlapping run_ids "
            f"{overlap_start}..{overlap_end - 1}. "
            f"Baseline range={baseline_start}..{baseline_start + baseline_count - 1}; "
            f"masked range={masked_start}..{masked_start + masked_count - 1}."
        )

    rows = []
    for run_id in range(overlap_start, overlap_end):
        baseline_index = run_id - baseline_start
        masked_index = run_id - masked_start
        row = {
            "run_id": run_id,
            "sample_index": run_id,
            "sample_number": run_id + 1,
            "baseline_index": baseline_index,
            "masked_index": masked_index,
        }
        for metric, _ in METRICS:
            baseline_value = float(metric_values[metric][0][baseline_index])
            masked_value = float(metric_values[metric][1][masked_index])
            row[f"baseline_{metric}"] = baseline_value
            row[f"masked_{metric}"] = masked_value
            row[f"delta_{metric}"] = masked_value - baseline_value
        rows.append(row)
    return rows


def _selected_metrics(metrics_arg):
    names = [name.strip().lower() for name in metrics_arg.split(",") if name.strip()]
    if not names:
        raise ValueError("--metrics must contain at least one metric name.")
    unknown = [name for name in names if name not in METRIC_BY_NAME]
    if unknown:
        raise ValueError(f"Unknown metric(s): {', '.join(unknown)}. Choose from: psnr,mse,ssim")
    return [(name, METRIC_BY_NAME[name]) for name in names]


def _long_dataframe(rows, metrics):
    records = []
    for row in rows:
        for metric, _ in metrics:
            for method in ("Baseline", "Masked"):
                records.append({
                    "sample_number": row["sample_number"],
                    "metric": metric,
                    "method": method,
                    "value": row[f"{method.lower()}_{metric}"],
                })
    return pd.DataFrame(records)


def _combined_long_dataframe(datasets, metrics):
    records = []
    for dataset_label, rows in datasets:
        for row in rows:
            for metric, _ in metrics:
                for method in ("Baseline", "Masked"):
                    records.append({
                        "dataset": dataset_label,
                        "sample_number": row["sample_number"],
                        "metric": metric,
                        "method": method,
                        "value": row[f"{method.lower()}_{metric}"],
                    })
    return pd.DataFrame(records)


def _outlier_dataframe(dataframe, group_cols, value_col="value"):
    outlier_parts = []
    for _, group in dataframe.groupby(group_cols, dropna=False):
        q1 = group[value_col].quantile(0.25)
        q3 = group[value_col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = group[(group[value_col] < lower) | (group[value_col] > upper)]
        if not outliers.empty:
            outlier_parts.append(outliers)
    if not outlier_parts:
        return dataframe.iloc[0:0].copy()
    return pd.concat(outlier_parts, ignore_index=True)


def _add_top_margin(ax, fraction=0.12):
    lower, upper = ax.get_ylim()
    span = upper - lower
    if span <= 0:
        return
    ax.set_ylim(lower, upper + span * fraction)


def _plot_method_outliers(ax, outliers, x, y, dodge=False):
    marker_by_method = {"Baseline": "X", "Masked": "D"}
    color_by_method = {"Baseline": "#1F4E79", "Masked": "#8B2E16"}
    for method, method_outliers in outliers.groupby("method", dropna=False):
        kwargs = {
            "data": method_outliers,
            "x": x,
            "y": y,
            "marker": marker_by_method.get(method, "X"),
            "edgecolor": "black",
            "linewidth": 0.8,
            "jitter": 0.06,
            "size": 6.0,
            "ax": ax,
        }
        if dodge:
            kwargs.update({
                "hue": "method",
                "hue_order": ["Baseline", "Masked"],
                "palette": color_by_method,
                "dodge": True,
            })
        else:
            kwargs["color"] = color_by_method.get(method, "black")
        sns.stripplot(**kwargs)


def _write_rows(rows, path):
    fieldnames = list(rows[0]) if rows else [
        "run_id",
        "sample_index",
        "sample_number",
        "baseline_index",
        "masked_index",
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


def _axes_for_metrics(metrics):
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]
    return fig, axes


def _plot(rows, path, title, metrics):
    dataframe = _long_dataframe(rows, metrics)
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = _axes_for_metrics(metrics)
    palette = {"Baseline": "#4C72B0", "Masked": "#DD8452"}
    for ax, (metric, ylabel) in zip(axes, metrics):
        metric_data = dataframe[dataframe["metric"] == metric]
        sns.violinplot(
            data=metric_data,
            x="method",
            y="value",
            hue="method",
            palette=palette,
            inner=None,
            cut=0,
            legend=False,
            ax=ax,
        )
        sns.boxplot(
            data=metric_data,
            x="method",
            y="value",
            width=0.22,
            showcaps=True,
            showfliers=False,
            boxprops={"facecolor": "none", "edgecolor": "black", "linewidth": 1.0},
            whiskerprops={"color": "black", "linewidth": 1.0},
            capprops={"color": "black", "linewidth": 1.0},
            medianprops={"color": "black", "linewidth": 1.2},
            ax=ax,
        )
        outliers = _outlier_dataframe(metric_data, ["method"])
        if not outliers.empty:
            _plot_method_outliers(ax, outliers, x="method", y="value")
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
    fig.suptitle(title)
    fig.tight_layout()
    if safe_savefig(fig, path, dpi=300, bbox_inches="tight"):
        print(f"Saved: {path}")
    plt.close(fig)


def _plot_combined(datasets, path, title, metrics):
    dataframe = _combined_long_dataframe(datasets, metrics)
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = _axes_for_metrics(metrics)
    palette = {"Baseline": "#4C72B0", "Masked": "#DD8452"}
    for ax, (metric, ylabel) in zip(axes, metrics):
        metric_data = dataframe[dataframe["metric"] == metric]
        sns.violinplot(
            data=metric_data,
            x="dataset",
            y="value",
            hue="method",
            palette=palette,
            split=True,
            inner=None,
            cut=0,
            ax=ax,
        )
        sns.boxplot(
            data=metric_data,
            x="dataset",
            y="value",
            hue="method",
            palette=palette,
            dodge=True,
            width=0.18,
            showfliers=False,
            boxprops={"facecolor": "none", "edgecolor": "black", "linewidth": 1.0},
            whiskerprops={"color": "black", "linewidth": 1.0},
            capprops={"color": "black", "linewidth": 1.0},
            medianprops={"color": "black", "linewidth": 1.2},
            ax=ax,
        )
        outliers = _outlier_dataframe(metric_data, ["dataset", "method"])
        if not outliers.empty:
            _plot_method_outliers(ax, outliers, x="dataset", y="value", dodge=True)
        handles, labels = ax.get_legend_handles_labels()
        if ax is axes[-1]:
            _add_top_margin(ax)
            ax.legend(
                handles[:2],
                labels[:2],
                title="Method",
                loc="upper right",
                ncol=2,
                frameon=True,
            )
        else:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
    fig.suptitle(title)
    fig.tight_layout()
    if safe_savefig(fig, path, dpi=300, bbox_inches="tight"):
        print(f"Saved: {path}")
    plt.close(fig)


def _plot_box(rows, path, title, metrics):
    dataframe = _long_dataframe(rows, metrics)
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = _axes_for_metrics(metrics)
    palette = {"Baseline": "#4C72B0", "Masked": "#DD8452"}
    for ax, (metric, ylabel) in zip(axes, metrics):
        metric_data = dataframe[dataframe["metric"] == metric]
        sns.boxplot(
            data=metric_data,
            x="method",
            y="value",
            palette=palette,
            width=0.5,
            showfliers=False,
            ax=ax,
        )
        outliers = _outlier_dataframe(metric_data, ["method"])
        if not outliers.empty:
            _plot_method_outliers(ax, outliers, x="method", y="value")
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
    fig.suptitle(title)
    fig.tight_layout()
    if safe_savefig(fig, path, dpi=300, bbox_inches="tight"):
        print(f"Saved: {path}")
    plt.close(fig)


def _delta_records(rows, metrics):
    records = []
    for row in rows:
        for metric, _ in metrics:
            label, value_fn = DELTA_LABELS[metric]
            records.append({"metric": label, "value": value_fn(row)})
    return records


def _plot_deltas(rows, path, title, metrics):
    records = _delta_records(rows, metrics)
    dataframe = pd.DataFrame(records)
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = _axes_for_metrics(metrics)
    for ax, metric in zip(axes, dataframe["metric"].unique()):
        metric_data = dataframe[dataframe["metric"] == metric]
        sns.violinplot(
            data=metric_data,
            x="metric",
            y="value",
            color="#55A868",
            inner=None,
            cut=0,
            ax=ax,
        )
        sns.boxplot(
            data=metric_data,
            x="metric",
            y="value",
            color="#55A868",
            width=0.22,
            showfliers=False,
            boxprops={"facecolor": "none", "edgecolor": "black", "linewidth": 1.0},
            whiskerprops={"color": "black", "linewidth": 1.0},
            capprops={"color": "black", "linewidth": 1.0},
            medianprops={"color": "black", "linewidth": 1.2},
            ax=ax,
        )
        outliers = _outlier_dataframe(metric_data, ["metric"])
        if not outliers.empty:
            sns.stripplot(
                data=outliers,
                x="metric",
                y="value",
                color="black",
                jitter=0.06,
                size=3.0,
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


def _plot_delta_box(rows, path, title, metrics):
    records = _delta_records(rows, metrics)
    dataframe = pd.DataFrame(records)
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = _axes_for_metrics(metrics)
    for ax, metric in zip(axes, dataframe["metric"].unique()):
        metric_data = dataframe[dataframe["metric"] == metric]
        sns.boxplot(
            data=metric_data,
            x="metric",
            y="value",
            color="#55A868",
            width=0.45,
            showfliers=False,
            ax=ax,
        )
        outliers = _outlier_dataframe(metric_data, ["metric"])
        if not outliers.empty:
            sns.stripplot(
                data=outliers,
                x="metric",
                y="value",
                color="black",
                jitter=0.06,
                size=3.0,
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
            f"baseline_index={row['baseline_index']}, "
            f"masked_index={row['masked_index']}, "
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
    parser.add_argument(
        "--paired_csv_path",
        default=None,
        help="Corrected per-sample CSV. When set, bypasses registry loading.",
    )
    parser.add_argument(
        "--paired_csv_paths",
        nargs="+",
        default=None,
        help=(
            "Multiple corrected per-sample CSVs for one combined dataset plot. "
            "Use with --dataset_labels."
        ),
    )
    parser.add_argument(
        "--dataset_labels",
        nargs="+",
        default=None,
        help="Dataset labels matching --paired_csv_paths.",
    )
    parser.add_argument(
        "--pair",
        nargs=3,
        action="append",
        metavar=("DATASET", "BASELINE_KEY", "MASKED_KEY"),
        help=(
            "Add one dataset from registry keys for the combined plot. "
            "Can be repeated: --pair LFW BASELINE_KEY MASKED_KEY --pair CIFAR-100 BASELINE_KEY MASKED_KEY"
        ),
    )
    parser.add_argument("--baseline_key")
    parser.add_argument("--masked_key")
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--output_prefix", default="paired_masking")
    parser.add_argument("--title", default=None)
    parser.add_argument(
        "--metrics",
        default="psnr,ssim",
        help="Comma-separated metrics to plot. Choose from psnr,mse,ssim.",
    )
    args = parser.parse_args()
    metrics = _selected_metrics(args.metrics)

    if args.paired_csv_paths and args.pair:
        parser.error("Use either --paired_csv_paths or --pair, not both")

    if args.paired_csv_paths:
        if args.dataset_labels and len(args.dataset_labels) != len(args.paired_csv_paths):
            parser.error("--dataset_labels must have the same length as --paired_csv_paths")
        labels = args.dataset_labels or [
            os.path.splitext(os.path.basename(path))[0] for path in args.paired_csv_paths
        ]
        datasets = [
            (label, _load_paired_rows(path))
            for label, path in zip(labels, args.paired_csv_paths)
        ]
        if not safe_makedirs(args.out_dir):
            return
        combined_path = os.path.join(args.out_dir, f"{args.output_prefix}_combined_violin.png")
        title = args.title or "Baseline vs masked reconstruction"
        _plot_combined(datasets, combined_path, title, metrics)
        return

    if args.pair:
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
        datasets = []
        for dataset_label, baseline_key, masked_key in args.pair:
            baseline_entry = _load_entry(baseline_registry, baseline_key, f"Baseline ({dataset_label})")
            masked_entry = _load_entry(masked_registry, masked_key, f"Masked ({dataset_label})")
            datasets.append((dataset_label, _paired_rows(baseline_entry, masked_entry)))
        if not safe_makedirs(args.out_dir):
            return
        combined_path = os.path.join(args.out_dir, f"{args.output_prefix}_combined_violin.png")
        title = args.title or "Baseline vs masked reconstruction"
        _plot_combined(datasets, combined_path, title, metrics)
        return

    baseline_entry = None
    if args.paired_csv_path:
        rows = _load_paired_rows(args.paired_csv_path)
    else:
        if not args.baseline_key or not args.masked_key:
            parser.error("--baseline_key and --masked_key are required unless --paired_csv_path is set")
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
    box_plot_path = os.path.join(args.out_dir, f"{args.output_prefix}_boxplot.png")
    delta_plot_path = os.path.join(args.out_dir, f"{args.output_prefix}_delta_violin.png")
    delta_box_plot_path = os.path.join(args.out_dir, f"{args.output_prefix}_delta_boxplot.png")
    _write_rows(rows, csv_path)
    baseline_args = baseline_entry.get("args", {}) if baseline_entry else {}
    title = args.title or (
        f"Baseline vs masked reconstruction: "
        f"{baseline_args.get('network', 'unknown')} / "
        f"{baseline_args.get('dataset', 'unknown')}"
    )
    _plot(rows, plot_path, title, metrics)
    _plot_box(rows, box_plot_path, title, metrics)
    _plot_deltas(rows, delta_plot_path, title, metrics)
    _plot_delta_box(rows, delta_box_plot_path, title, metrics)
    _print_extremes(rows)


if __name__ == "__main__":
    main()
