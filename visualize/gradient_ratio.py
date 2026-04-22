import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

networks = [
    "LeNet", "LeNet-bigger", "MediumCNN",
    "ResNet-18", "ResNet-50", "ResNet-152",
    "VGG-11", "VGG-16", "VGG-19",
]
params = [13426, 46890, 1060132, 11227812, 23712932, 58348708, 129176036, 134670244, 139979940]

mnist_unknowns = 784
cifar_unknowns = 3072

ratio_mnist = [p / mnist_unknowns for p in params]
ratio_cifar  = [p / cifar_unknowns for p in params]

fig, ax = plt.subplots(figsize=(8, 5))
n = len(networks)
y = np.arange(n)
bar_h = 0.35

color_mnist = "#4C72B0"
color_cifar  = "#DD8452"

ax.barh(y - bar_h/2, ratio_cifar, bar_h,
        label="CIFAR-10 / CIFAR-100 / LFW  (3,072 unknowns)",
        color=color_cifar, zorder=3)
ax.barh(y + bar_h/2, ratio_mnist, bar_h,
        label="MNIST  (784 unknowns)",
        color=color_mnist, zorder=3)

ax.axvline(x=1, color="black", linewidth=1.2, linestyle="--", zorder=4,
           label="Threshold  (ratio = 1)")

ax.set_xscale("log")
ax.set_yticks(y)
ax.set_yticklabels(networks, fontsize=11)
ax.set_xlabel("Gradient-to-pixel ratio  (log scale)", fontsize=11)
ax.set_title("Gradient entries per input unknown across architectures", fontsize=12)

ax.xaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, _: f"{x:,.0f}" if x >= 1 else f"{x:.2f}"
))

ax.legend(fontsize=9, loc="lower right")
ax.grid(axis="x", which="major", linestyle=":", linewidth=0.6, alpha=0.7, zorder=0)
ax.set_axisbelow(True)

ax.axhline(y=2.5, color="grey", linewidth=0.6, linestyle="-", alpha=0.5)
ax.axhline(y=5.5, color="grey", linewidth=0.6, linestyle="-", alpha=0.5)

ax.text(1.01, (0 + 2)/2, "Custom\nCNNs",  transform=ax.get_yaxis_transform(),
        ha="left", va="center", fontsize=8.5, color="grey")
ax.text(1.01, (3 + 5)/2, "ResNet\nfamily", transform=ax.get_yaxis_transform(),
        ha="left", va="center", fontsize=8.5, color="grey")
ax.text(1.01, (6 + 8)/2, "VGG\nfamily",  transform=ax.get_yaxis_transform(),
        ha="left", va="center", fontsize=8.5, color="grey")

plt.tight_layout()
out = "/home/mathias/GitHub/bachelor_thesis/Pictures/gradient_ratio.pdf"
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print("Saved:", out)
