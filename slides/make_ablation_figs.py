"""Generate the ablation *example* schematics in one identical box-grid format.

A shared header/legend template (title, monospace mode line, rounded EXAMPLE
pill, swatch legend, fonts, canvas) frames every figure, and a single
``draw_box`` renders the layer-group boxes for all of them, so the figures read
as a matched set. The only thing that varies between modes is how much of each
box is masked:

  * leave_one_layer_out_example.png  -> prefix_topfrac_entries_layer (f=1.0,
        binary): each masked run drops one WHOLE named group (box fully ✕).
  * gradsize_topfrac_entries_layer_example.png -> gradsize_topfrac_entries_layer:
        a FRACTION of every box is masked (kept part blue, masked part grey ✕);
        ranking is within each parameter tensor (weight/bias separately).
  * masking_modes_combined_example.png -> both, stacked: 2 rows leave-one-out +
        2 rows fractional.

All are schematics / examples and flagged as such. Run:
    python slides/make_ablation_figs.py
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, Rectangle

# ----------------------------------------------------------------- palette ---
DTU_RED = "#990000"
BLUE = "#2f6db0"
BLUE_EDGE = "#1f4e79"
GREY = "#d9d9d9"
GREY_EDGE = "#b5b5b5"
INK = "#222222"
MUTE = "#6b6b6b"
BADGE_FILL = "#fbeaea"
MONO = "DejaVu Sans Mono"
NAME_FX = [pe.withStroke(linewidth=2.2, foreground="#16314d")]
plt.rcParams.update({"font.family": "DejaVu Sans"})

ASSETS = Path(__file__).resolve().parent / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)
FIGSIZE = (12.4, 7.0)

LAYERS = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"]
BW, BH, GAP, PITCH, X0 = 1.0, 0.62, 0.16, 1.05, 2.05
BLOCK_RIGHT = X0 + (len(LAYERS) - 1) * (BW + GAP) + BW


# ============================================================ shared chrome ===
def base_figure(title, mode, desc):
    fig = plt.figure(figsize=FIGSIZE)
    fig.text(0.045, 0.945, title, fontsize=18, fontweight="bold", color=INK,
             ha="left", va="center")
    fig.text(0.045, 0.905, mode, fontsize=12, color=DTU_RED, ha="left",
             va="center", family=MONO)
    fig.text(0.045, 0.862, desc, fontsize=11, color=MUTE, ha="left", va="top",
             style="italic", linespacing=1.3)
    fig.text(0.897, 0.931, "EXAMPLE", fontsize=12.5, fontweight="bold",
             color=DTU_RED, ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.55,rounding_size=1.4",
                       facecolor=BADGE_FILL, edgecolor=DTU_RED, linewidth=1.8))
    return fig


def swatch(fig, x, y, face, edge, label, cross=False):
    fig.patches.append(Rectangle((x, y), 0.022, 0.030, transform=fig.transFigure,
                                 clip_on=False, linewidth=1.2,
                                 edgecolor=edge, facecolor=face))
    if cross:
        for y0, y1 in [(0.006, 0.024), (0.024, 0.006)]:
            fig.add_artist(plt.Line2D([x + 0.005, x + 0.017], [y + y0, y + y1],
                           transform=fig.transFigure, color=DTU_RED, lw=1.6))
    fig.text(x + 0.030, y + 0.015, label, fontsize=10.5, color=INK,
             ha="left", va="center")


# ----------------------------------------------------- one layer-group box ---
def draw_box(ax, x, y, kept, name=None):
    """kept in [0,1]: 1 = fully retained, 0 = fully masked, else partial."""
    full_masked = kept <= 1e-4
    full_kept = kept >= 1 - 1e-4
    edge = DTU_RED if full_masked else BLUE_EDGE
    lw = 1.6 if full_masked else 1.2
    box = FancyBboxPatch((x, y), BW, BH,
                         boxstyle="round,pad=0.02,rounding_size=0.10",
                         facecolor=(GREY if full_masked else BLUE),
                         edgecolor=edge, linewidth=lw)
    ax.add_patch(box)

    if not full_masked and not full_kept:          # partial: grey the masked part
        gx = x + kept * BW
        grey = Rectangle((gx, y), (1 - kept) * BW, BH, facecolor=GREY,
                         edgecolor="none")
        ax.add_patch(grey); grey.set_clip_path(box)
        ax.plot([gx, gx], [y + 0.06, y + BH - 0.06], color=DTU_RED,
                ls=(0, (2, 2)), lw=1.1)
        ca, cb = gx, x + BW
    elif full_masked:
        ca, cb = x, x + BW
    else:
        ca = None

    if ca is not None:                              # red cross over masked region
        pad = 0.14
        for ya, yb in [(y + pad, y + BH - pad), (y + BH - pad, y + pad)]:
            ln, = ax.plot([ca + pad, cb - pad], [ya, yb], color=DTU_RED, lw=2.2)
            ln.set_clip_path(box)

    ax.add_patch(FancyBboxPatch((x, y), BW, BH,                     # crisp outline
                 boxstyle="round,pad=0.02,rounding_size=0.10",
                 facecolor="none", edgecolor=edge, linewidth=lw))
    if name and not full_masked:
        ax.text(x + BW / 2, y + BH / 2, name, ha="center", va="center",
                fontsize=9, color="white", fontweight="bold", path_effects=NAME_FX)


def place_row(ax, y, kept_list, label, desc, base=False):
    ax.text(X0 - 0.35, y + BH / 2 + 0.09, label, ha="right", va="center",
            fontsize=12, color=INK, fontweight=("bold" if base else "normal"))
    if desc:
        ax.text(X0 - 0.35, y + BH / 2 - 0.20, desc, ha="right", va="center",
                fontsize=9, color=(MUTE if base else DTU_RED), style="italic")
    for i, name in enumerate(LAYERS):
        draw_box(ax, X0 + i * (BW + GAP), y, kept_list[i], name)


def body_axes(fig, ylo, yhi):
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.71]); ax.axis("off")
    ax.set_xlim(-2.7, BLOCK_RIGHT + 0.2)
    ax.set_ylim(ylo, yhi)
    return ax


# =================================================== fig 1: leave-one-out =====
def fig_leave_one_out():
    fig = base_figure(
        "Leave-one-layer-out ablation",
        "mode: prefix_topfrac_entries_layer   (retention fraction = 1.0 → binary)",
        "Each masked run excludes one entire named gradient group while retaining "
        "all others; the baseline keeps every\ngroup. Masking is binary at the "
        "group level. The excluded layers shown here are illustrative.")
    swatch(fig, 0.045, 0.770, BLUE, BLUE_EDGE, "retained — matched in the loss")
    swatch(fig, 0.40, 0.770, GREY, GREY_EDGE, "masked — excluded", cross=True)

    rows = [([1] * 7, "Baseline", "retain all groups", True),
            (_drop("conv1"), "Masked run", "e.g. exclude conv1", False),
            (_drop("layer2"), "Masked run", "e.g. exclude layer2", False),
            (_drop("fc"), "Masked run", "e.g. exclude fc", False)]
    top = len(rows) * PITCH
    ax = body_axes(fig, -0.9, top + 0.15)
    ax.text(X0 - 0.05, top - 0.02, "gradient layer-groups →", fontsize=9.5,
            color=MUTE, style="italic", ha="left")
    for r, (kept, lab, desc, base) in enumerate(rows):
        place_row(ax, (len(rows) - 1 - r) * PITCH, kept, lab, desc, base)
    ax.text((X0 + BLOCK_RIGHT) / 2, -0.52,
            "… one masked run for every group (bn1, layer1, layer3, …)",
            fontsize=10.5, color=MUTE, ha="center", style="italic")
    return _save(fig, "leave_one_layer_out_example.png")


# ============================= fig 2: gradsize fractional (box format) ========
def fig_gradsize_boxes():
    fig = base_figure(
        "Magnitude-based fractional masking",
        "mode: gradsize_topfrac_entries_layer",
        "Instead of removing whole groups, a FRACTION f of every group is kept: "
        "entries are ranked by |gradient| and the\nlowest-magnitude ones are "
        "masked (grey ✕). The sweep varies f. Ranking is within each parameter "
        "tensor; fractions shown are examples.")
    swatch(fig, 0.045, 0.770, BLUE, BLUE_EDGE, "kept — largest-|gradient| entries")
    swatch(fig, 0.40, 0.770, GREY, GREY_EDGE, "masked — smallest-|gradient| entries",
           cross=True)

    rows = [([1.0] * 7, "Baseline", "f = 1.00  (0% masked)", True),
            ([0.75] * 7, "f = 0.75", "mask 25% of each box", False),
            ([0.50] * 7, "f = 0.50", "mask 50% of each box", False),
            ([0.25] * 7, "f = 0.25", "mask 75% of each box", False)]
    top = len(rows) * PITCH
    ax = body_axes(fig, -0.9, top + 0.15)
    ax.text(X0 - 0.05, top - 0.02, "gradient layer-groups →", fontsize=9.5,
            color=MUTE, style="italic", ha="left")
    for r, (kept, lab, desc, base) in enumerate(rows):
        place_row(ax, (len(rows) - 1 - r) * PITCH, kept, lab, desc, base)
    ax.text((X0 + BLOCK_RIGHT) / 2, -0.52,
            "fraction applied within each parameter tensor (weight & bias "
            "separately); shown per group here for comparison",
            fontsize=10, color=MUTE, ha="center", style="italic")
    return _save(fig, "gradsize_topfrac_entries_layer_example.png")


# ===================================== fig 3: combined (2 rows + 2 rows) ======
def fig_combined():
    fig = base_figure(
        "Two ways to mask the same gradient",
        "prefix_topfrac_entries_layer (binary)   ·   gradsize_topfrac_entries_layer (fractional)",
        "Both act on the same named layer-groups. Leave-one-out removes a whole "
        "group at a time; the fractional mode keeps\nonly the top-magnitude "
        "fraction of entries inside every group. Example configurations shown.")
    swatch(fig, 0.045, 0.770, BLUE, BLUE_EDGE, "retained / kept")
    swatch(fig, 0.30, 0.770, GREY, GREY_EDGE,
           "masked — whole group, or a fraction of each group", cross=True)

    ax = body_axes(fig, -0.95, 5.55)
    # section 1: leave-one-out
    ax.text(X0 - 0.35, 5.18, "Leave-one-layer-out", ha="right", fontsize=11.5,
            fontweight="bold", color=INK)
    ax.text(X0 + 0.05, 5.18, "prefix_topfrac_entries_layer — binary",
            ha="left", fontsize=9.5, color=DTU_RED, family=MONO, va="center")
    place_row(ax, 4.30, [1] * 7, "Baseline", "retain all groups", base=True)
    place_row(ax, 3.25, _drop("layer2"), "Masked run", "e.g. exclude layer2")

    ax.plot([X0 - 1.05, BLOCK_RIGHT], [2.78, 2.78], color="#d8d8d8", lw=1.0)

    # section 2: fractional
    ax.text(X0 - 0.35, 2.45, "Fraction kept per group", ha="right", fontsize=11.5,
            fontweight="bold", color=INK)
    ax.text(X0 + 0.05, 2.45, "gradsize_topfrac_entries_layer — fractional",
            ha="left", fontsize=9.5, color=DTU_RED, family=MONO, va="center")
    place_row(ax, 1.55, [0.50] * 7, "f = 0.50", "mask 50% of each box")
    place_row(ax, 0.50, [0.25] * 7, "f = 0.25", "mask 75% of each box")

    ax.text((X0 + BLOCK_RIGHT) / 2, -0.55,
            "fractional masking ranks entries within each parameter tensor "
            "(weight & bias separately); shown per group for comparison",
            fontsize=10, color=MUTE, ha="center", style="italic")
    return _save(fig, "masking_modes_combined_example.png")


# --------------------------------------------------------------- helpers -----
def _drop(layer):
    return [0 if L == layer else 1 for L in LAYERS]


def _save(fig, name):
    out = ASSETS / name
    fig.savefig(out, dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    for builder in (fig_leave_one_out, fig_gradsize_boxes, fig_combined):
        print("wrote", builder())
