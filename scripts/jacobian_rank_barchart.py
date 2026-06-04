"""
Grouped bar chart: minimum gradient entries to reach full Jacobian rank (d=3072),
two bars per network (layer_spread vs topk_abs).
Style matches the ablation forest plots (sns whitegrid, paper context).

Output: jacobian_rank_barchart.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
import pandas as pd

sns.set_theme(style="whitegrid", context="paper")

networks     = ["LeNet", "ResNet-18", "ResNet-50", "VGG-11", "VGG-13"]
layer_spread = [6_000,   9_500,      9_500,       7_500,    6_500]
topk_abs     = [6_500,   6_500,      10_000,      46_500,   30_000]

df = pd.DataFrame({
    "Network":        networks * 2,
    "Select mode":    ["layer_spread"] * len(networks) + ["topk_abs"] * len(networks),
    "Gradient entries": layer_spread + topk_abs,
})

palette = {
    "layer_spread": sns.color_palette()[0],
    "topk_abs":     sns.color_palette()[1],
}

fig, ax = plt.subplots(figsize=(9, 5))

sns.barplot(
    data=df,
    x="Network",
    y="Gradient entries",
    hue="Select mode",
    palette=palette,
    ax=ax,
)

ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax.set_ylabel("Gradient entries to full rank")
ax.set_xlabel("Model")
ax.set_title(
    "Minimum gradient entries to reach full Jacobian rank ($d = 3{,}072$) on CIFAR-100",
)
ax.legend(title="Select mode", framealpha=0.9, loc="upper left")
ax.grid(True, axis="x")

# value labels
x_coords = [p.get_x() + p.get_width() / 2 for p in ax.patches]
heights   = [p.get_height() for p in ax.patches]
ymax = max(heights) * 1.18
ax.set_ylim(0, ymax)
for xc, h in zip(x_coords, heights):
    if h > 0:
        ax.text(xc, h + ymax * 0.008, f"{int(h):,}", ha="center", va="bottom", fontsize=7.5)

plt.tight_layout()
plt.savefig("jacobian_rank_barchart.png", dpi=300, bbox_inches="tight")
print("Saved jacobian_rank_barchart.png")
