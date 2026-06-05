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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from stable_ginv.io import safe_makedirs, safe_savefig, safe_write

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
                "topfrac": float(row["topfrac"]),
                "n_reconstructed": int(row["n_reconstructed"]),
                "n_total": int(row["n_total"]),
                "avg_mse": float(row["avg_mse"]),
                "median_mse": float(row["median_mse"]),
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


def _table_rows(rows, include_baseline):
    table_rows = rows if include_baseline else [row for row in rows if row["source"] != "baseline"]
    return sorted(
        table_rows,
        key=lambda row: (
            row["network"].lower(),
            row["source"] != "baseline",
            row["pct_masked"],
        ),
    )


def _format_pct(value):
    return f"{value:.0f}"


def _write_combined_table(rows, csv_path, tex_path, include_baseline):
    table_rows = _table_rows(rows, include_baseline)
    fields = [
        "network",
        "dataset",
        "source",
        "topfrac",
        "pct_masked",
        "n_reconstructed",
        "n_total",
        "avg_mse",
        "median_mse",
    ]

    def _write_csv(f):
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(table_rows)

    if safe_write(csv_path, _write_csv, newline=""):
        print(f"Saved: {csv_path}")

    def _write_tex(f):
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{llcccc}\n")
        f.write("\\toprule\n")
        f.write(
            "Network & Condition & Masked (\\%) & Reconstructed & "
            "Avg. \\textsc{MSE} & Median \\textsc{MSE} \\\\\n"
        )
        f.write("\\midrule\n")
        previous_network = None
        for row in table_rows:
            network = _network_label(row["network"])
            if previous_network is not None and previous_network != network:
                f.write("\\midrule\n")
            previous_network = network
            condition = "Unmasked" if row["source"] == "baseline" else "Masked"
            f.write(
                f"{network} & {condition} & {_format_pct(row['pct_masked'])} & "
                f"${row['n_reconstructed']}/{row['n_total']}$ & "
                f"{row['avg_mse']:.4f} & {row['median_mse']:.4f} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write(
            "\\caption{Masking sweep reconstruction counts using "
            "\\textsc{MSE} $\\leq 0.01$ as the reconstruction threshold.}\n"
        )
        f.write("\\label{tab:combined_masking_sweep}\n")
        f.write("\\end{table}\n")

    if safe_write(tex_path, _write_tex):
        print(f"Saved: {tex_path}")


def _plot(rows, out_path, include_baseline):
    masked_rows = [row for row in rows if row["source"] == "masked"]
    baseline_rows = [row for row in rows if row["source"] == "baseline"]
    if not masked_rows:
        raise ValueError("No masked rows found in the supplied summary CSVs.")

    networks = sorted({row["network"] for row in masked_rows}, key=str.lower)
    palette = dict(zip(networks, sns.color_palette("tab10", n_colors=len(networks))))
    datasets = sorted({row["dataset"] for row in rows})
    n_total_values = sorted({row["n_total"] for row in rows})

    fig, ax = plt.subplots(figsize=(12, 5))

    for network in networks:
        network_rows = sorted(
            [row for row in masked_rows if row["network"] == network],
            key=lambda row: row["pct_masked"],
        )
        if include_baseline:
            network_rows = (
                [row for row in baseline_rows if row["network"] == network]
                + network_rows
            )
        sns.lineplot(
            x=[row["pct_masked"] for row in network_rows],
            y=[row["n_reconstructed"] for row in network_rows],
            marker="o",
            linewidth=1.8,
            color=palette[network],
            label=_network_label(network),
            ax=ax,
        )

    if include_baseline:
        for network in networks:
            network_rows = [row for row in baseline_rows if row["network"] == network]
            if not network_rows:
                continue
            ax.scatter(
                [row["pct_masked"] for row in network_rows],
                [row["n_reconstructed"] for row in network_rows],
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
    max_pct = max(row["pct_masked"] for row in rows)
    x_max = int(np.ceil(max_pct / 10.0) * 10)
    y_max = int(np.ceil(n_total_label / 10.0) * 10)
    ax.set_ylim(-0.5, y_max + 0.5)
    ax.set_xlim(left=-2, right=x_max + 2)
    ax.set_xticks(np.arange(0, x_max + 1, 10))
    ax.set_yticks(np.arange(0, y_max + 1, 10))
    ax.legend(title="Network")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir and not safe_makedirs(out_dir):
        return
    if safe_savefig(fig, out_path, dpi=150):
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
    parser.add_argument(
        "--table_csv_path",
        default=None,
        help="Output CSV table path. Defaults to <out_path stem>_table.csv.",
    )
    parser.add_argument(
        "--table_tex_path",
        default=None,
        help="Output LaTeX table path. Defaults to <out_path stem>_table.tex.",
    )
    args = parser.parse_args()

    rows = []
    for path in args.summary_csv:
        rows.extend(_read_summary(path))
    include_baseline = not args.no_baseline
    _plot(rows, args.out_path, include_baseline=include_baseline)
    out_stem, _ = os.path.splitext(args.out_path)
    _write_combined_table(
        rows,
        args.table_csv_path or f"{out_stem}_table.csv",
        args.table_tex_path or f"{out_stem}_table.tex",
        include_baseline=include_baseline,
    )


if __name__ == "__main__":
    main()
