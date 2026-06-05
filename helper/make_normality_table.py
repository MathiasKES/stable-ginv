"""
Emit LaTeX Shapiro-Wilk normality tables from precomputed experiment results.

For each withheld-layer masking configuration, a 2x2 facet of subtables is
written (rows = pretraining, columns = grad loss). Each subtable lists the
W statistic, p-value, and the normality decision at alpha for the paired
(masked - unmasked) PSNR/MSE/SSIM differences.

The W/p values are parsed from the ``*_normality`` strings stored in an
experiment-results JSON (e.g. ``results/exp_results_resnet_manual.json``);
only ``method == "masked"`` entries carry them. Output goes to a single
``.tex`` file intended to be \\input into a document that loads
``booktabs``, ``subcaption``, and ``amssymb``.

Example:
    python helper/make_normality_table.py \
        --results_json results/exp_results_resnet_manual.json \
        --network resnet50 --n 30 \
        --out results/normality_table.tex
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from functions.io_utils import safe_makedirs


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

METRIC_ORDER = ["psnr", "mse", "ssim"]
METRIC_LABELS = {"psnr": "PSNR", "mse": "MSE", "ssim": "SSIM"}

LOSS_ORDER = ["l2", "cos"]
LOSS_LABELS = {"l2": r"$\ell_2$", "cos": "cosine"}
PRETRAIN_ORDER = [False, True]
PRETRAIN_LABELS = {False: "no pretraining", True: "pretrained"}

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


def load_table(results_json, layer_order):
    """Build table[layer][(loss, pretrained)][metric] = (W, p)."""
    with open(results_json) as handle:
        data = json.load(handle)

    table = {}
    for entry in data.values():
        if entry.get("method") != "masked":
            continue
        layer = withheld_layer(entry.get("prefixes", ""), layer_order)
        if layer is None:
            continue
        key = (entry.get("grad_loss", ""), bool(entry.get("pretrained", False)))
        for metric in METRIC_ORDER:
            parsed = parse_normality(entry.get(f"{metric}_normality", ""))
            if parsed is None:
                continue
            table.setdefault(layer, {}).setdefault(key, {})[metric] = parsed
    return table


def fmt_p(p_value):
    return r"$<0.0001$" if p_value < 0.0001 else f"${p_value:.4f}$"


def render_subtable(cell, loss, pretrained, alpha):
    """Render one 0.48\\textwidth subtable for a (loss, pretrained) cell."""
    caption = f"{LOSS_LABELS[loss]}, {PRETRAIN_LABELS[pretrained]}"
    lines = [
        r"  \begin{subtable}[t]{0.48\textwidth}",
        r"    \centering",
        f"    \\caption{{{caption}}}",
        r"    \begin{tabular}{lccc}",
        r"    \toprule",
        r"    Paired difference & $W$ & $p$-value & Normal at $\alpha=" + f"{alpha:g}$ \\\\",
        r"    \midrule",
    ]
    for metric in METRIC_ORDER:
        label = f"\\textsc{{{METRIC_LABELS[metric]}}}"
        if cell and metric in cell:
            w_stat, p_value = cell[metric]
            mark = r"\checkmark" if p_value >= alpha else r"\texttimes"
            lines.append(
                f"    {label} & ${w_stat:.4f}$ & {fmt_p(p_value)} & {mark} \\\\"
            )
        else:
            lines.append(f"    {label} & -- & -- & -- \\\\")
    lines += [
        r"    \bottomrule",
        r"    \end{tabular}",
        r"  \end{subtable}",
    ]
    return "\n".join(lines)


def render_layer_grid(network, layer, cells, n_samples, alpha):
    """Render one 2x2 facet (table float) for a single withheld layer."""
    caption = (
        f"Shapiro--Wilk normality test on the paired (masked $-$ unmasked) "
        f"differences for the {network} \\texttt{{{layer}}}-withheld masking "
        f"configuration on CIFAR-100 ($n={n_samples}$), across the four "
        f"(loss $\\times$ pretraining) settings. Normality is rejected "
        f"(\\texttimes) when $p<{alpha:g}$."
    )
    parts = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \small",
        f"  \\caption{{{caption}}}",
        f"  \\label{{tab:shapiro_{network}_{layer.replace('.', '')}}}",
    ]
    for r_idx, pretrained in enumerate(PRETRAIN_ORDER):
        for c_idx, loss in enumerate(LOSS_ORDER):
            cell = cells.get((loss, pretrained))
            parts.append(render_subtable(cell, loss, pretrained, alpha))
            if c_idx == 0:
                parts.append(r"  \hfill")
        if r_idx == 0:
            parts.append(r"  \vspace{1em}")
    parts.append(r"\end{table}")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Emit LaTeX Shapiro-Wilk normality tables (2x2 facet per layer)."
    )
    parser.add_argument(
        "--results_json",
        default=os.path.join("results", "exp_results_resnet_manual.json"),
    )
    parser.add_argument("--network", default="resnet50", choices=sorted(ABLATION_LAYERS))
    parser.add_argument("--n", type=int, default=30, help="Sample size for the caption.")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out", default=os.path.join("results", "normality_table.tex"))
    args = parser.parse_args()

    layer_order = ABLATION_LAYERS[args.network]
    table = load_table(args.results_json, layer_order)
    if not table:
        raise SystemExit(f"No masked normality records found in {args.results_json}")

    header = (
        "% Auto-generated by helper/make_normality_table.py -- do not edit by hand.\n"
        "% Requires: \\usepackage{booktabs}, \\usepackage{subcaption}, "
        "\\usepackage{amssymb}\n"
    )
    grids = [
        render_layer_grid(args.network, layer, table[layer], args.n, args.alpha)
        for layer in layer_order if layer in table
    ]

    safe_makedirs(os.path.dirname(args.out) or ".")
    with open(args.out, "w") as handle:
        handle.write(header + "\n" + "\n\n".join(grids) + "\n")
    print(f"Wrote {args.out} ({len(grids)} layer grids)")


if __name__ == "__main__":
    main()
