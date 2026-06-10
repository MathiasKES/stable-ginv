"""Real per-metric normality figure for the methods deck.

Reads the paired baseline/masked per-image metrics from
slides/data/normality_resnet18_layer4.json, forms the per-image differences
d = masked - baseline (exactly what the thesis feeds to Shapiro-Wilk, see
stats/paired.py and Appendix C / Table C.2), and plots their distributions with
the real W and p annotated. MSE differences are right-skewed and fail the test;
PSNR and SSIM differences are symmetric and pass — which is why normality is
assessed per metric.

Run:  python slides/make_normality_fig.py
Out:  slides/assets/normality_per_metric.png
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

DTU_RED = "#990000"; BLUE = "#2f6db0"; GREEN = "#1b7a3d"
INK = "#222222"; MUTE = "#6b6b6b"
plt.rcParams.update({"font.family": "DejaVu Sans"})

HERE = Path(__file__).resolve().parent
DATA = HERE / "normality_resnet18_layer4.json"

PANELS = [
    ("MSE difference", "best_mse_list", DTU_RED, "bounded near 0 → right-skewed"),
    ("PSNR difference (dB)", "best_psnr_list", BLUE, "log of MSE → ~symmetric"),
    ("SSIM difference", "best_ssim_list", GREEN, "~symmetric"),
]


def _draw_panel(ax, base, mask, label, key, col, note, show_note=True):
    diff = np.array(mask[key]) - np.array(base[key])
    w, p = stats.shapiro(diff)
    sk = stats.skew(diff)
    ax.hist(diff, bins=11, density=True, color=col, alpha=0.28,
            edgecolor=col, linewidth=0.5)
    xs = np.linspace(diff.min(), diff.max(), 300)
    ax.plot(xs, stats.gaussian_kde(diff)(xs), color=col, lw=2.2)
    ax.plot(xs, stats.norm.pdf(xs, diff.mean(), diff.std(ddof=1)),
            color=INK, lw=1.4, ls="--")
    ax.axvline(0, color="#999999", lw=0.8, ls=":")
    verdict = "NON-NORMAL" if p < 0.05 else "normal"
    vcol = DTU_RED if p < 0.05 else GREEN
    pstr = "p < 0.001" if p < 0.001 else f"p = {p:.2f}"
    ax.set_title(label, fontsize=13, fontweight="bold", color=INK, pad=8)
    ax.text(0.96, 0.95, f"Shapiro–Wilk\nW = {w:.3f}\n{pstr}\nskew = {sk:+.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9.5,
            color=INK, linespacing=1.35)
    ax.text(0.96, 0.55, verdict, transform=ax.transAxes, ha="right",
            va="top", fontsize=11, fontweight="bold", color=vcol)
    if show_note:
        ax.text(0.5, -0.15, note, transform=ax.transAxes, ha="center", va="top",
                fontsize=9.5, color=MUTE, style="italic")
    ax.set_yticks([]); ax.tick_params(labelsize=9)
    ax.spines[["top", "right", "left"]].set_visible(False)


def render(compact=False):
    """compact=False -> standalone titled figure; True -> panels-only for a slide."""
    d = json.loads(DATA.read_text())
    cfg, base, mask = d["config"], d["baseline"], d["masked"]

    if compact:
        fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5))
        plt.subplots_adjust(left=0.03, right=0.99, top=0.86, bottom=0.16, wspace=0.13)
        for ax, spec in zip(axes, PANELS):
            _draw_panel(ax, base, mask, *spec, show_note=False)
        fig.text(0.5, 0.04,
                 "Per-image differences (masked − baseline), n = 30 · dashed = "
                 "fitted normal · reproduces Appendix C (Table C.2, layer4)",
                 fontsize=9, color=MUTE, ha="center", style="italic")
        out = HERE / "assets" / "normality_per_metric_slide.png"
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6))
        plt.subplots_adjust(left=0.045, right=0.985, top=0.72, bottom=0.22, wspace=0.13)
        for ax, spec in zip(axes, PANELS):
            _draw_panel(ax, base, mask, *spec)
        fig.text(0.045, 0.92, "Why normality is tested per metric", fontsize=15,
                 fontweight="bold", color=INK)
        fig.text(0.045, 0.85,
                 f"{cfg['network']} · {cfg['dataset']} · L-BFGS / L2 · no pretrain · "
                 f"mask = {cfg['masked_layer']} · per-image differences "
                 f"(masked − baseline), n = {cfg['n']}",
                 fontsize=10.5, color=MUTE)
        fig.text(0.045, 0.05,
                 "Dashed = fitted normal reference; dotted line = 0.  MSE differences "
                 "are right-skewed and Shapiro–Wilk rejects normality; PSNR and SSIM "
                 "differences are symmetric and pass.",
                 fontsize=9.5, color=MUTE, style="italic")
        fig.text(0.045, 0.015,
                 "Reproduces Appendix C (Table C.2, layer4): p(MSE) < 0.001, "
                 "p(PSNR) = 0.98, p(SSIM) = 0.80.", fontsize=9, color=MUTE)
        out = HERE / "assets" / "normality_per_metric.png"

    fig.savefig(out, dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    render(compact=False)
    render(compact=True)
