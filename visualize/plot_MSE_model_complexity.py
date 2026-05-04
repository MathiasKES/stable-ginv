import math
import matplotlib.pyplot as plt
from scipy.stats import t

# INPUT DATA
models_mnist_adamw = [
    {"name": "LeNet", "params": 15826,    "avg_PSNR": 51.69226, "std_PSNR": 7.46322, "n": 30},
    {"name": "LeNet_bigger", "params": 51210,    "avg_PSNR": 45.24061, "std_PSNR": 10.13012, "n": 30},
    {"name": "MediumCNN", "params": 322762,    "avg_PSNR": 48.19717, "std_PSNR": 11.71229, "n": 30},
    {"name": "BiggerCNN", "params": 1899338,    "avg_PSNR": 37.1661, "std_PSNR": 12.2871, "n": 30},
    {"name": "resnet18", "params": 11689512,    "avg_PSNR": 11.54265, "std_PSNR": 1.73189, "n": 30},
    {"name": "resnet34", "params": 21797672,    "avg_PSNR": 9.69364, "std_PSNR": 1.72308, "n": 30},
    {"name": "resnet50", "params": 25557032,    "avg_PSNR": 7.88037, "std_PSNR": 1.15285, "n": 30},
    {"name": "resnet101", "params": 44549160,    "avg_PSNR": 7.32658, "std_PSNR": 0.69318, "n": 30},
    {"name": "resnet152", "params": 60192808,    "avg_PSNR": 7.28744, "std_PSNR": 0.91452, "n": 30},
]

models_cifar100_adamw = [
    {"name": "LeNet", "params": 15826,    "avg_PSNR": 18.77469, "std_PSNR": 3.43323, "n": 30},
    {"name": "LeNet_bigger", "params": 51210,    "avg_PSNR": 21.42534, "std_PSNR": 4.46527, "n": 30},
    {"name": "MediumCNN", "params": 322762,    "avg_PSNR": 30.51882, "std_PSNR": 10.96286, "n": 30},
    {"name": "BiggerCNN", "params": 1899338,    "avg_PSNR": 29.54332, "std_PSNR": 10.67741, "n": 30},
    {"name": "resnet18", "params": 11689512,    "avg_PSNR": 15.59134, "std_PSNR": 2.1862, "n": 30},
    {"name": "resnet34", "params": 21797672,    "avg_PSNR": 15.39698, "std_PSNR": 2.366, "n": 30},
    {"name": "resnet50", "params": 25557032,    "avg_PSNR": 13.66543, "std_PSNR": 2.48372, "n": 30},
    {"name": "resnet101", "params": 44549160,    "avg_PSNR": 12.55754, "std_PSNR": 2.26314, "n": 30},
    {"name": "resnet152", "params": 60192808,    "avg_PSNR": 13.39473, "std_PSNR": 2.3074, "n": 30},
]
plt.figure(figsize=(9, 6))

for model in models_cifar100:
    avg = model["avg_PSNR"]
    std = model["std_PSNR"]
    n = model["n"]

    # 95% confidence interval half-width
    t_crit = t.ppf(0.975, df=n-1)
    ci_half = t_crit * std / math.sqrt(n)

    plt.errorbar(
        model["params"],
        avg,
        yerr=ci_half,
        fmt='o',
        capsize=5,
        label=f'{model["name"]} (n={n})'
    )

    plt.annotate(
        f'{avg:.2f} ± {ci_half:.2f}',
        (model["params"], avg),
        textcoords="offset points",
        xytext=(8, 8)
    )

plt.xlabel("Number of Parameters")
plt.ylabel("Average PSNR")
plt.title("Model Complexity vs Reconstruction Error (95% CI) on CIFAR100")
plt.xscale("log")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/model_complexity_vs_psnr_CIFAR100.png", dpi=300, bbox_inches="tight")
plt.show()