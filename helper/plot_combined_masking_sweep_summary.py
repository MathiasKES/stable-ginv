"""
Overlay multiple masking-sweep summary CSVs in one plot.

Example:
    python helper/plot_combined_masking_sweep_summary.py \
        results/masking_sweep_summary_LeNet_cifar100_threshold_0p01.csv \
        results/masking_sweep_summary_resnet18_cifar100_threshold_0p01.csv \
        results/masking_sweep_summary_vgg13_cifar100_threshold_0p01.csv \
        --out_path results/combined_masking_sweep_cifar100_threshold_0p01.png
"""
import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from functions.io_utils import safe_makedirs, safe_savefig

sns.set_theme(style="whitegrid")


def _read_summary(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "network": row["network"],
                "dataset": row["dataset"],
                "source": row["source"],
                "pct_masked": float(row["pct_masked"]),
                "n_reconstructed": int(row["n_reconstructed"]),
                "n_total": int(row["n_total"]),
                "path": path,
            })
    return rows


def _network_label(network):
    labels = {
        "resnet18": "ResNet18",
        "vgg13": "VGG13",
        "LeNet": "LeNet",
    }
    return labels.get(network, network)


def _plot(rows, out_path, include_baseline):
    masked_rows = [row for row in rows if row["source"] == "masked"]
    baseline_rows = [row for row in rows if row["source"] == "baseline"]
    if not masked_rows:
        raise ValueError("No masked rows found in the supplied summary CSVs.")

    networks = sorted({row["network"] for row in masked_rows}, key=str.lower)
    palette = dict(zip(networks, sns.color_palette("tab10", n_colors=len(networks))))
    datasets = sorted({row["dataset"] for row in rows})
    n_total_values = sorted({row["n_total"] for row in rows})

    fig, ax = plt.subplots(figsize=(8, 4.8))

    for network in networks:
        network_rows = sorted(
            [row for row in masked_rows if row["network"] == network],
            key=lambda row: row["pct_masked"],
        )
        ax.plot(
            [row["pct_masked"] for row in network_rows],
            [row["n_reconstructed"] for row in network_rows],
            marker="o",
            linewidth=2,
            color=palette[network],
            label=_network_label(network),
        )

        if include_baseline:
            for row in baseline_rows:
                if row["network"] != network:
                    continue
                ax.scatter(
                    row["pct_masked"],
                    row["n_reconstructed"],
                    marker="D",
                    s=55,
                    facecolors="white",
                    edgecolors=palette[network],
                    linewidths=1.6,
                    zorder=5,
                )

    dataset_label = datasets[0] if len(datasets) == 1 else "mixed datasets"
    n_total_label = n_total_values[0] if len(n_total_values) == 1 else max(n_total_values)
    ax.set_title(f"Masking sweep comparison: {dataset_label}")
    ax.set_xlabel("Gradient entries masked (%)")
    ax.set_ylabel("Images reconstructed (MSE <= 0.01)")
    ax.set_ylim(-0.5, n_total_label + 0.5)
    ax.set_xlim(left=-2)
    ax.legend(title="Network")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir and not safe_makedirs(out_dir):
        return
    if safe_savefig(fig, out_path, dpi=200):
        print(f"Saved: {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_csv", nargs="+", help="masking_sweep_summary_*.csv files")
    parser.add_argument(
        "--out_path",
        default="results/combined_masking_sweep_summary.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--no_baseline",
        action="store_true",
        help="Do not draw baseline markers from source=baseline rows.",
    )
    args = parser.parse_args()

    rows = []
    for path in args.summary_csv:
        rows.extend(_read_summary(path))
    _plot(rows, args.out_path, include_baseline=not args.no_baseline)


if __name__ == "__main__":
    main()
