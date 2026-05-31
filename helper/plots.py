import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = [
    # layer, config, ci_low, ci_high

    # L-BFGS no pretrain
    ("conv1",  "L-BFGS\nNo pretrain", -2.97, -1.39),
    ("bn1",    "L-BFGS\nNo pretrain", -0.10,  0.50),
    ("layer1", "L-BFGS\nNo pretrain", -1.70, -0.85),
    ("layer2", "L-BFGS\nNo pretrain", -1.39, -0.60),
    ("layer3", "L-BFGS\nNo pretrain", -0.83, -0.17),
    ("layer4", "L-BFGS\nNo pretrain", -0.20,  0.34),

    # L-BFGS pretrain
    ("conv1",  "L-BFGS\nPretrained", -4.35, -2.44),
    ("bn1",    "L-BFGS\nPretrained", -1.60,  0.34),
    ("layer1", "L-BFGS\nPretrained", -1.46,  0.54),
    ("layer2", "L-BFGS\nPretrained", -2.58, -0.69),
    ("layer3", "L-BFGS\nPretrained", -2.76, -1.05),
    ("layer4", "L-BFGS\nPretrained", -4.40, -1.86),

    # Signed AdamW no pretrain
    ("conv1",  "Signed AdamW\nNo pretrain", -1.06,  0.46),
    ("bn1",    "Signed AdamW\nNo pretrain", -0.49,  0.12),
    ("layer1", "Signed AdamW\nNo pretrain", -1.66, -1.02),
    ("layer2", "Signed AdamW\nNo pretrain", -1.54, -0.90),
    ("layer3", "Signed AdamW\nNo pretrain", -1.04, -0.27),
    ("layer4", "Signed AdamW\nNo pretrain", -0.61,  0.02),

    # Signed AdamW pretrained
    ("conv1",  "Signed AdamW\nPretrained", -3.06, -2.36),
    ("bn1",    "Signed AdamW\nPretrained", -0.51,  0.01),
    ("layer1", "Signed AdamW\nPretrained", -1.22, -0.64),
    ("layer2", "Signed AdamW\nPretrained", -1.22, -0.44),
    ("layer3", "Signed AdamW\nPretrained", -0.65,  0.01),
    ("layer4", "Signed AdamW\nPretrained",  0.01,  1.07),
]

# =========================
# ResNet18 (old)
# =========================
resnet18_data = data


# =========================
# VGG13 (new)
# =========================
vgg13_data = [

    # L-BFGS no pretrain
    ("features.0",  "L-BFGS\nNo pretrain", -0.03,  0.46),
    ("features.2",  "L-BFGS\nNo pretrain",  0.16,  0.70),
    ("features.5",  "L-BFGS\nNo pretrain",  0.30,  0.96),
    ("features.7",  "L-BFGS\nNo pretrain",  0.30,  1.03),
    ("features.10", "L-BFGS\nNo pretrain",  0.36,  0.93),
    ("features.12", "L-BFGS\nNo pretrain",  0.28,  0.98),
    ("features.15", "L-BFGS\nNo pretrain",  0.36,  1.00),
    ("features.17", "L-BFGS\nNo pretrain",  0.12,  0.61),
    ("features.20", "L-BFGS\nNo pretrain",  0.09,  0.71),
    ("features.22", "L-BFGS\nNo pretrain", -0.02,  0.84),
    ("classifier",  "L-BFGS\nNo pretrain",  0.20,  0.90),

    # L-BFGS pretrained
    ("features.0",  "L-BFGS\nPretrained", -1.19, -0.35),
    ("features.2",  "L-BFGS\nPretrained", -0.86, -0.05),
    ("features.5",  "L-BFGS\nPretrained", -0.86, -0.15),
    ("features.7",  "L-BFGS\nPretrained", -0.53,  0.40),
    ("features.10", "L-BFGS\nPretrained", -0.91,  0.12),
    ("features.12", "L-BFGS\nPretrained", -0.48,  0.18),
    ("features.15", "L-BFGS\nPretrained", -0.82,  0.07),
    ("features.17", "L-BFGS\nPretrained", -0.46,  0.28),
    ("features.20", "L-BFGS\nPretrained", -1.16, -0.22),
    ("features.22", "L-BFGS\nPretrained", -1.59, -0.29),
    ("classifier",  "L-BFGS\nPretrained", -1.07, -0.06),

    # Signed AdamW no pretrain
    ("features.0",  "Signed AdamW\nNo pretrain",  1.03,  2.77),
    ("features.2",  "Signed AdamW\nNo pretrain",  0.43,  1.44),
    ("features.5",  "Signed AdamW\nNo pretrain",  0.31,  1.31),
    ("features.7",  "Signed AdamW\nNo pretrain", -0.04,  0.83),
    ("features.10", "Signed AdamW\nNo pretrain", -0.42,  0.47),
    ("features.12", "Signed AdamW\nNo pretrain", -0.57,  0.29),
    ("features.15", "Signed AdamW\nNo pretrain", -0.67,  0.19),
    ("features.17", "Signed AdamW\nNo pretrain", -0.76,  0.19),
    ("features.20", "Signed AdamW\nNo pretrain", -0.87,  0.13),
    ("features.22", "Signed AdamW\nNo pretrain", -0.84, -0.04),
    ("classifier",  "Signed AdamW\nNo pretrain", -0.24,  0.69),

    # Signed AdamW pretrained
    ("features.0",  "Signed AdamW\nPretrained", -0.48,  0.03),
    ("features.2",  "Signed AdamW\nPretrained", -0.32,  0.05),
    ("features.5",  "Signed AdamW\nPretrained", -0.35, -0.11),
    ("features.7",  "Signed AdamW\nPretrained", -0.19,  0.04),
    ("features.10", "Signed AdamW\nPretrained", -0.25,  0.08),
    ("features.12", "Signed AdamW\nPretrained", -0.30,  0.07),
    ("features.15", "Signed AdamW\nPretrained", -0.28,  0.04),
    ("features.17", "Signed AdamW\nPretrained", -0.15,  0.18),
    ("features.20", "Signed AdamW\nPretrained", -0.47,  0.10),
    ("features.22", "Signed AdamW\nPretrained", -0.37,  0.09),
    ("classifier",  "Signed AdamW\nPretrained", -0.19,  0.21),
]

vgg11_data = [
    # L-BFGS no pretrain
    ("features.0",  "L-BFGS\nNo pretrain", -0.49,  0.63),
    ("features.3",  "L-BFGS\nNo pretrain", -0.07,  1.35),
    ("features.6",  "L-BFGS\nNo pretrain",  0.16,  1.98),
    ("features.8",  "L-BFGS\nNo pretrain", -0.07,  0.82),
    ("features.11", "L-BFGS\nNo pretrain",  0.06,  1.54),
    ("features.13", "L-BFGS\nNo pretrain", -0.29,  0.59),
    ("features.16", "L-BFGS\nNo pretrain", -0.53,  0.03),
    ("classifier",  "L-BFGS\nNo pretrain",  0.27,  2.34),

    # L-BFGS pretrained
    ("features.0",  "L-BFGS\nPretrained", -1.83,  0.82),
    ("features.3",  "L-BFGS\nPretrained", -1.52,  0.41),
    ("features.6",  "L-BFGS\nPretrained", -1.15,  1.10),
    ("features.8",  "L-BFGS\nPretrained", -1.56,  0.86),
    ("features.11", "L-BFGS\nPretrained", -1.55,  0.41),
    ("features.13", "L-BFGS\nPretrained", -1.70,  1.21),
    ("features.16", "L-BFGS\nPretrained", -1.29, -0.03),
    ("classifier",  "L-BFGS\nPretrained", -1.44,  0.87),

    # Signed AdamW no pretrain
    ("features.0",  "Signed AdamW\nNo pretrain",  0.16,  3.82),
    ("features.3",  "Signed AdamW\nNo pretrain", -2.18, -0.90),
    ("features.6",  "Signed AdamW\nNo pretrain", -2.49, -0.83),
    ("features.8",  "Signed AdamW\nNo pretrain", -3.53, -1.95),
    ("features.11", "Signed AdamW\nNo pretrain", -2.76, -1.48),
    ("features.13", "Signed AdamW\nNo pretrain", -3.82, -1.98),
    ("features.16", "Signed AdamW\nNo pretrain", -4.76, -2.61),
    ("classifier",  "Signed AdamW\nNo pretrain", -1.92,  0.01),

    # Signed AdamW pretrained
    ("features.0",  "Signed AdamW\nPretrained", -0.65, -0.14),
    ("features.3",  "Signed AdamW\nPretrained", -0.57,  0.03),
    ("features.6",  "Signed AdamW\nPretrained", -0.48,  0.14),
    ("features.8",  "Signed AdamW\nPretrained", -0.56,  0.08),
    ("features.11", "Signed AdamW\nPretrained", -0.39,  0.19),
    ("features.13", "Signed AdamW\nPretrained", -0.42,  0.27),
    ("features.16", "Signed AdamW\nPretrained", -0.82,  0.02),
    ("classifier",  "Signed AdamW\nPretrained", -0.63,  0.15),
]

#df = pd.DataFrame(data, columns=["layer", "config", "ci_low", "ci_high"])

#df = pd.DataFrame(vgg13_data, columns=["layer", "config", "ci_low", "ci_high"])

df = pd.DataFrame(vgg11_data, columns=["layer", "config", "ci_low", "ci_high"])

# midpoint of CI as point estimate
df["mean"] = (df["ci_low"] + df["ci_high"]) / 2

# asymmetric error bars
df["err_low"] = df["mean"] - df["ci_low"]
df["err_high"] = df["ci_high"] - df["mean"]

#layer_order = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]
# layer_order = [
#     "features.0",
#     "features.2",
#     "features.5",
#     "features.7",
#     "features.10",
#     "features.12",
#     "features.15",
#     "features.17",
#     "features.20",
#     "features.22",
#     "classifier",
# ]
layer_order = [
    "features.0",
    "features.3",
    "features.6",
    "features.8",
    "features.11",
    "features.13",
    "features.16",
    "classifier",
]
sns.set_theme(style="whitegrid", context="paper")

g = sns.FacetGrid(
    df,
    col="config",
    col_wrap=2,
    height=5,
    aspect=1.2,
    sharex=True,
    sharey=True
)

def forestplot(data, **kwargs):
    ax = plt.gca()

    data = data.copy()
    data["layer"] = pd.Categorical(
        data["layer"],
        categories=layer_order,
        ordered=True
    )

    data = data.sort_values("layer", ascending=False)

    y = range(len(data))

    ax.errorbar(
        x=data["mean"],
        y=y,
        xerr=[data["err_low"], data["err_high"]],
        fmt="o",
        capsize=4,
        linewidth=1.5,
        markersize=5
    )

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1
    )

    ax.set_yticks(list(y))
    ax.set_yticklabels(data["layer"])

    ax.set_xlabel(r"$\Delta$PSNR (dB)")
    ax.set_ylabel("Masked layer")

g.map_dataframe(forestplot)

g.set_titles("{col_name}")

# plt.suptitle(
#     "ResNet18 Ablation Study\nPSNR Difference (Masked − Baseline)",
#     y=1.03
# )

# plt.suptitle(
#     "VGG13 Ablation Study\nPSNR Difference (Masked − Baseline)",
#     y=1.03
# )
plt.suptitle(
    "VGG11 Ablation Study\nPSNR Difference (Masked − Baseline)",
    y=1.03
)

plt.tight_layout()

plt.savefig("vgg11_ablation_forestplot.pdf", bbox_inches="tight")
plt.savefig("vgg11_ablation_forestplot.png", dpi=300, bbox_inches="tight")

# plt.savefig(
#     "resnet18_ablation_forestplot.pdf",
#     bbox_inches="tight"
# )

# plt.savefig(
#     "resnet18_ablation_forestplot.png",
#     dpi=300,
#     bbox_inches="tight"
# )

# plt.savefig("vgg13_ablation_forestplot.pdf", bbox_inches="tight")
# plt.savefig("vgg13_ablation_forestplot.png", dpi=300, bbox_inches="tight")

plt.show()