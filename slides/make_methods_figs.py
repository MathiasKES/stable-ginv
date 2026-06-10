"""Schematic figures for the Methods deck (gradient masking + normality test).

Both figures are conceptual schematics, not plots of experimental data; the
normality figure in particular illustrates the *shape* argument from the thesis
(MSE right-skewed, PSNR symmetrised by the log) and is labelled as such.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy import stats

OUT = Path(__file__).resolve().parent / "assets"
OUT.mkdir(parents=True, exist_ok=True)

DTU_RED = "#990000"
BLUE = "#2f6db0"
GREY = "#d9d9d9"
INK = "#222222"
plt.rcParams.update({"font.family": "DejaVu Sans"})

# ----------------------------------------------------------- masking diagram --
def masking_diagram():
    layers = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"]
    masked_each = [None, "conv1", "layer2", "fc"]  # rows: baseline + 3 examples
    row_labels = ["Baseline\n(retain all)", "Mask conv1", "Mask layer2", "Mask fc"]

    fig, ax = plt.subplots(figsize=(9.4, 4.3))
    n = len(layers)
    bw, bh, gap = 1.0, 0.62, 0.18
    for r, (masked, rlab) in enumerate(zip(masked_each, row_labels)):
        y = (len(masked_each) - 1 - r) * (bh + 0.55)
        ax.text(-0.35, y + bh / 2, rlab, ha="right", va="center",
                fontsize=11, color=INK, fontweight="bold")
        for i, lname in enumerate(layers):
            x = i * (bw + gap)
            is_masked = (lname == masked)
            face = GREY if is_masked else BLUE
            box = FancyBboxPatch((x, y), bw, bh,
                                 boxstyle="round,pad=0.02,rounding_size=0.08",
                                 linewidth=1.2,
                                 edgecolor=(DTU_RED if is_masked else "#1f4e79"),
                                 facecolor=face)
            ax.add_patch(box)
            if not is_masked:
                ax.text(x + bw / 2, y + bh / 2, lname, ha="center", va="center",
                        fontsize=9.5, color="white", fontweight="bold")
            if is_masked:
                ax.plot([x + 0.12, x + bw - 0.12], [y + 0.12, y + bh - 0.12],
                        color=DTU_RED, lw=2.2)
                ax.plot([x + 0.12, x + bw - 0.12], [y + bh - 0.12, y + 0.12],
                        color=DTU_RED, lw=2.2)
    ax.set_xlim(-2.6, n * (bw + gap))
    ax.set_ylim(-0.3, len(masked_each) * (bh + 0.55))
    ax.axis("off")
    ax.text(-2.55, len(masked_each) * (bh + 0.55) - 0.12,
            "Named gradient layer-groups", fontsize=10.5, style="italic",
            color="#555555")
    fig.tight_layout()
    fig.savefig(OUT / "masking_diagram.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)

# ------------------------------------------------ normality shape schematic ---
def normality_schematic():
    rng = np.random.default_rng(7)
    # right-skewed positive quantity standing in for per-image MSE differences
    mse = rng.lognormal(mean=-3.2, sigma=0.9, size=4000)
    psnr = -10 * np.log10(mse + 1e-12)  # log transform -> symmetrised

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))
    for ax, data, title, col in [
        (axes[0], mse, "MSE differences\n(bounded at 0, right-skewed)", DTU_RED),
        (axes[1], psnr, "PSNR differences\n(log compresses the tail → symmetric)", BLUE),
    ]:
        ax.hist(data, bins=40, color=col, alpha=0.30, density=True,
                edgecolor=col, linewidth=0.4)
        xs = np.linspace(data.min(), data.max(), 300)
        kde = stats.gaussian_kde(data)
        ax.plot(xs, kde(xs), color=col, lw=2.2)
        # normal reference
        mu, sd = data.mean(), data.std()
        ax.plot(xs, stats.norm.pdf(xs, mu, sd), color=INK, lw=1.4, ls="--")
        ax.set_title(title, fontsize=11, color=INK)
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(labelsize=9)
    axes[1].plot([], [], color=INK, ls="--", lw=1.4, label="normal reference")
    axes[1].legend(fontsize=9, frameon=False, loc="upper right")
    fig.suptitle("Schematic: why the Shapiro–Wilk test is applied per metric",
                 fontsize=12, fontweight="bold", color=INK, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "normality_schematic.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)

masking_diagram()
normality_schematic()
print("wrote", OUT / "masking_diagram.png", "and", OUT / "normality_schematic.png")
