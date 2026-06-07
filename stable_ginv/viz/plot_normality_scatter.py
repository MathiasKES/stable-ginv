"""
Plot Shapiro-Wilk normality results (W statistic and p-value) for masked
ablation experiments as a 2x2 facet scatter grid.

The W statistic and p-value are read from the precomputed ``*_normality``
strings stored in an experiment-results JSON (e.g.
``results/exp_results_resnet_manual.json``), which look like::

    "psnr_normality": "NON-NORMAL (W=0.9035, p=0.0102)"
    "mse_normality":  "normal (W=0.9308, p=0.0515)"

Only ``method == "masked"`` entries carry normality data; ``idlg`` baselines
have empty fields and are skipped. Each masked entry withholds exactly one
network layer group from the mask; that withheld layer becomes its row.

Entries are grouped by ``(grad_loss, pretrained)`` into a 2x2 facet grid
(rows = pretrained, columns = grad_loss). Each facet shows two columns of
points: the W statistic (oriented 1 -> 0, so more-normal sits left) and the
p-value (0 -> 1). A dashed line marks the significance level ``alpha`` and the
rejection region ``p < alpha`` (where normality is rejected) is shaded red.

Example:
    python -m stable_ginv.viz.plot_normality_scatter \
        --results_json results/exp_results_resnet_manual.json \
        --network resnet50 \
        --out_dir results/normality \
        --output_prefix resnet50_normality_scatter
"""
import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns

from stable_ginv.io import safe_makedirs, safe_savefig


ABLATION_LAYERS = {
    "lenet": ["body.0", "body.2", "body.4", "fc"],
    "resnet18": ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"],
    "resnet50": ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"],
    "vgg13": [
        "features.0", "features.2", "features.5", "features.7",
        "features.10", "features.12", "features.15", "features.17",
        "features.20", "features.22", "classifier",
    ],
    "vgg11": [
        "features.0", "features.3", "features.6", "features.8",
        "features.11", "features.13", "features.16", "classifier",
    ],
}

# Metrics drawn per layer row, in stacked order (top -> bottom within a row).
METRIC_ORDER = ["psnr", "mse", "ssim"]
METRIC_LABELS = {"psnr": "PSNR", "mse": "MSE", "ssim": "SSIM"}

# Grad-loss variants share each panel, distinguished by marker shape.
LOSS_ORDER = ["l2", "cos"]
LOSS_MARKERS = {"l2": "*", "cos": "o"}

# Six points per layer row: grouped by metric (color), split by loss (shape).
# Each point gets a distinct vertical offset so none overlap.
METRIC_BASE = {"psnr": -0.27, "mse": 0.0, "ssim": 0.27}
LOSS_SUBOFFSET = {"l2": -0.07, "cos": 0.07}

NORMALITY_RE = re.compile(r"W=\s*([0-9.eE+-]+)\s*,\s*p=\s*([0-9.eE+-]+)")


def parse_normality(text):
    """Extract (W, p) from a normality string, or None if unavailable."""
    if not text:
        return None
    match = NORMALITY_RE.search(text)
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def withheld_layer(prefixes, layer_order):
    """Return the single layer group absent from a masked entry's prefixes."""
    kept = {part.split(":")[0] for part in prefixes.split(",") if part}
    missing = [layer for layer in layer_order if layer not in kept]
    return missing[0] if len(missing) == 1 else None


def load_records(results_json, layer_order):
    """Parse masked entries into per (group, layer, metric) normality records."""
    with open(results_json) as handle:
        data = json.load(handle)

    records = []
    for entry in data.values():
        if entry.get("method") != "masked":
            continue
        layer = withheld_layer(entry.get("prefixes", ""), layer_order)
        if layer is None:
            continue
        group = (entry.get("grad_loss", ""), bool(entry.get("pretrained", False)))
        for metric in METRIC_ORDER:
            parsed = parse_normality(entry.get(f"{metric}_normality", ""))
            if parsed is None:
                continue
            w_stat, p_value = parsed
            records.append(
                {
                    "group": group,
                    "layer": layer,
                    "metric": metric,
                    "W": w_stat,
                    "p": p_value,
                }
            )
    return records


def plot_normality(records, layer_order, alpha, network, out_path):
    """Render the merged (W, p) scatter grid and save it.

    Layout is 2 rows (pretrained) x 2 columns (W, p). Both grad-loss variants
    share each panel: colour encodes the metric, marker shape encodes the loss.
    """
    sns.set_theme(style="whitegrid", context="paper")
    colors = dict(zip(METRIC_ORDER, sns.color_palette("colorblind", len(METRIC_ORDER))))

    pretrain_states = [False, True]
    nrows = len(pretrain_states)

    w_values = [rec["W"] for rec in records]
    w_floor = (min(w_values) - 0.02) if w_values else 0.0

    n_layers = len(layer_order)
    layer_index = {layer: i for i, layer in enumerate(layer_order)}

    frame = pd.DataFrame(records)
    frame["grad_loss"] = frame["group"].map(lambda g: g[0])
    frame["pretrained"] = frame["group"].map(lambda g: g[1])
    frame["y"] = (
        frame["layer"].map(layer_index)
        + frame["metric"].map(METRIC_BASE)
        + frame["grad_loss"].map(LOSS_SUBOFFSET)
    )

    fig, axes = plt.subplots(
        nrows, 2,
        figsize=(7.5, 1.0 + 0.8 * n_layers),
        sharey=True,
        squeeze=False,
    )

    for row, pretrained in enumerate(pretrain_states):
        sub = frame[frame["pretrained"] == pretrained]
        for col, value_col in enumerate(("W", "p")):
            ax = axes[row][col]
            if not sub.empty:
                sns.scatterplot(
                    data=sub, x=value_col, y="y",
                    hue="metric", hue_order=METRIC_ORDER, palette=colors,
                    style="grad_loss", style_order=LOSS_ORDER, markers=LOSS_MARKERS,
                    s=60, edgecolor="white", linewidth=0.5, legend=False,
                    zorder=3, ax=ax,
                )

            ax.set_ylim(n_layers - 0.5, -0.5)
            ax.set_yticks(range(n_layers))
            ax.set_ylabel("")
            # Drop horizontal gridlines; keep vertical ones for reading off x.
            ax.yaxis.grid(False)
            ax.xaxis.grid(True)
            # Thin separators at the midpoints between adjacent layer rows.
            for sep in range(n_layers - 1):
                ax.axhline(sep + 0.5, color="0.85", linewidth=0.8, zorder=1)

            if value_col == "W":
                # Oriented 1 -> floor, so more-normal sits on the left.
                ax.set_xlim(1.01, w_floor)
            else:
                ax.set_xlim(-0.02, 1.02)
                ax.axvspan(-0.02, alpha, color="red", alpha=0.12, zorder=0)
                ax.axvline(alpha, color="red", linestyle="--", linewidth=1.2, zorder=2)

            if row == 0:
                ax.set_title("W statistic" if value_col == "W" else "p-value",
                             fontsize=12, fontweight="semibold")
            if row == nrows - 1:
                ax.set_xlabel("← more normal" if value_col == "W" else "p-value")
            else:
                ax.set_xlabel("")

        axes[row][0].set_yticklabels(layer_order)
        axes[row][0].set_ylabel("Pretrained" if pretrained else "Not pretrained",
                                fontsize=12, fontweight="semibold")

    metric_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=7,
               markerfacecolor=colors[m], markeredgecolor="white",
               label=METRIC_LABELS[m])
        for m in METRIC_ORDER
    ]
    loss_handles = [
        Line2D([0], [0], marker=LOSS_MARKERS[loss], linestyle="", markersize=9,
               markerfacecolor="0.3", markeredgecolor="0.3", label=loss)
        for loss in LOSS_ORDER
    ]
    alpha_handle = Line2D([0], [0], color="red", linestyle="--", linewidth=1.2,
                          label=f"alpha = {alpha:g}")
    legend_handles = metric_handles + loss_handles + [alpha_handle]
    fig.legend(handles=legend_handles, loc="upper center", ncol=len(legend_handles),
               frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"{network}: Shapiro-Wilk normality of paired metric differences",
                 y=1.05, fontsize=13, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    safe_savefig(fig, out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    """Plot Shapiro-Wilk normality scatter grid for masked ablation experiments.

    Reads a ``*_normality`` field from each ``method == "masked"`` entry in
    ``--results_json``, determines which network layer group was withheld for
    that entry, and accumulates (W, p) pairs for every (metric, grad_loss,
    pretrained) combination.

    Calls :func:`plot_normality` to render a 2 x 2 facet scatter grid
    (rows = pretrained state, columns = W statistic / p-value) and saves the
    figure to ``<out_dir>/<output_prefix>.png``.
    """
    parser = argparse.ArgumentParser(
        description="Plot Shapiro-Wilk normality (W, p) scatter grid for masked "
                    "ablation experiments."
    )
    parser.add_argument(
        "--results_json",
        default=os.path.join("results", "exp_results_resnet_manual.json"),
        help="Experiment-results JSON keyed by registry with *_normality fields.",
    )
    parser.add_argument("--network", default="resnet50", choices=sorted(ABLATION_LAYERS))
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level for the rejection region.")
    parser.add_argument("--out_dir", default=os.path.join("results", "normality"))
    parser.add_argument("--output_prefix", default="normality_scatter")
    args = parser.parse_args()

    layer_order = ABLATION_LAYERS[args.network]
    records = load_records(args.results_json, layer_order)
    if not records:
        raise SystemExit(f"No masked normality records found in {args.results_json}")

    safe_makedirs(args.out_dir)
    out_path = os.path.join(args.out_dir, f"{args.output_prefix}.png")
    plot_normality(records, layer_order, args.alpha, args.network, out_path)
    print(f"Wrote {out_path} ({len(records)} points)")


if __name__ == "__main__":
    main()
