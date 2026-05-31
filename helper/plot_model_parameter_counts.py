"""Plot parameter counts for the model architectures used in experiments."""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from functions.io_utils import safe_savefig
from helper.Network import get_model


MODEL_NAMES = ["LeNet", "resnet18", "resnet50", "vgg11", "vgg13"]
DISPLAY_NAMES = {
    "LeNet": "LeNet",
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "vgg11": "VGG-11",
    "vgg13": "VGG-13",
}


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def count_parameters_without_classifier(model):
    excluded_parameters = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or name == "classifier" or name.startswith("classifier."):
            excluded_parameters.update(id(parameter) for parameter in module.parameters())
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if id(parameter) not in excluded_parameters
    )


def get_parameter_counts(channel, num_classes, input_size):
    counts = []
    for network in MODEL_NAMES:
        model = get_model(
            network,
            channel=channel,
            num_classes=num_classes,
            input_size=input_size,
        )
        counts.append((
            DISPLAY_NAMES[network],
            count_parameters(model),
            count_parameters_without_classifier(model),
        ))
    return counts


def save_parameter_count_plot(counts, output_path, title, log_scale=False):
    labels = [label for label, _ in counts]
    counts_millions = [count / 1_000_000 for _, count in counts]

    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        x=np.asarray(labels),
        y=np.asarray(counts_millions),
        ax=ax,
        color=sns.color_palette()[0],
    )

    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel("Parameters (millions)")
    if log_scale:
        ax.set_yscale("log")
    ax.bar_label(
        ax.containers[0],
        labels=[f"{count:,}" for _, count in counts],
        padding=3,
        fontsize=8,
    )
    ax.margins(y=0.12)
    fig.tight_layout()

    saved = safe_savefig(fig, output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    if saved:
        print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--input_size", type=int, nargs=2, default=(32, 32), metavar=("H", "W"))
    parser.add_argument("--output_log", default="results/model_parameter_counts_log.png")
    parser.add_argument(
        "--output_without_classifier",
        default="results/model_parameter_counts_without_classifier.png",
    )
    parser.add_argument(
        "--output_without_classifier_log",
        default="results/model_parameter_counts_without_classifier_log.png",
    )
    args = parser.parse_args()

    counts = get_parameter_counts(args.channel, args.num_classes, tuple(args.input_size))
    total_counts = [(label, total) for label, total, _ in counts]
    backbone_counts = [(label, backbone) for label, _, backbone in counts]
    for label, total, backbone in counts:
        print(f"{label}: {total:,} total; {backbone:,} without classifier or fully connected layers")
    save_parameter_count_plot(
        total_counts,
        args.output_log,
        title="Model Parameter Counts",
        log_scale=True,
    )
    save_parameter_count_plot(
        backbone_counts,
        args.output_without_classifier,
        title="Model Parameter Counts Without Classifier or Fully Connected Layers",
    )
    save_parameter_count_plot(
        backbone_counts,
        args.output_without_classifier_log,
        title="Model Parameter Counts Without Classifier or Fully Connected Layers",
        log_scale=True,
    )


if __name__ == "__main__":
    main()
