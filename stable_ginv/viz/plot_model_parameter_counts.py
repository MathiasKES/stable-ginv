"""Plot parameter counts for the model architectures used in experiments."""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch.nn as nn

from stable_ginv.io import safe_savefig
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
    """Count the total number of scalar parameters in a model.

    Parameters
    ----------
    model : torch.nn.Module
        The model whose parameters are to be counted.

    Returns
    -------
    int
        Total number of elements across all parameter tensors.
    """
    return sum(parameter.numel() for parameter in model.parameters())


def count_parameters_without_classifier(model):
    """Count parameters excluding all ``nn.Linear`` and classifier layers.

    Any parameter that belongs to an ``nn.Linear`` module, the ``classifier``
    module, or any submodule whose name starts with ``"classifier."`` is
    excluded from the count.

    Parameters
    ----------
    model : torch.nn.Module
        The model whose backbone (non-classifier) parameters are counted.

    Returns
    -------
    int
        Total number of elements in parameters not belonging to any linear
        or classifier module.
    """
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
    """Instantiate each model in ``MODEL_NAMES`` and return its parameter counts.

    Parameters
    ----------
    channel : int
        Number of input image channels (e.g. 3 for RGB).
    num_classes : int
        Number of output classes used to size the classifier head.
    input_size : tuple of int
        Spatial resolution ``(H, W)`` passed to :func:`get_model`.

    Returns
    -------
    list of tuple
        Each element is ``(display_name, total_params, backbone_params)``
        where *display_name* is the human-readable network name from
        :data:`DISPLAY_NAMES`, *total_params* is the full parameter count,
        and *backbone_params* excludes classifier / fully connected layers.
    """
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


def save_parameter_count_plot(counts, output_path):
    """Render and save a grouped bar chart of model parameter counts.

    Draws two bars per model: total parameter count and backbone-only count
    (without classifier / fully connected layers).  The y-axis uses a log
    scale; each bar is labelled with the exact integer count.

    Parameters
    ----------
    counts : list of tuple
        Sequence of ``(display_name, total_params, backbone_params)`` as
        returned by :func:`get_parameter_counts`.
    output_path : str
        Destination path for the PNG figure (e.g.
        ``"results/model_parameter_counts_log.png"``).
    """
    labels = [label for label, _, _ in counts]
    total_counts = [total for _, total, _ in counts]
    backbone_counts = [backbone for _, _, backbone in counts]
    total_counts_millions = [count / 1_000_000 for count in total_counts]
    backbone_counts_millions = [count / 1_000_000 for count in backbone_counts]
    positions = np.arange(len(labels))
    width = 0.36

    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    palette = sns.color_palette()
    total_bars = ax.bar(
        positions - width / 2,
        total_counts_millions,
        width=width,
        color=palette[0],
        label="Total parameters",
    )
    backbone_bars = ax.bar(
        positions + width / 2,
        backbone_counts_millions,
        width=width,
        color=palette[1],
        label="Without classifier or fully connected layers",
    )

    ax.set_title("Model Parameter Counts")
    ax.set_xlabel("Model")
    ax.set_ylabel("Parameters (millions)")
    ax.set_xticks(positions, labels)
    ax.set_yscale("log")
    backbone_min = min(backbone_counts_millions)
    overall_max = max(total_counts_millions + backbone_counts_millions)
    ax.set_ylim(bottom=backbone_min * 0.35, top=overall_max * 3.0)
    ax.legend(loc="upper left")
    ax.bar_label(
        total_bars,
        labels=[f"{count:,}" for count in total_counts],
        padding=8,
        fontsize=7,
        rotation=25,
    )
    ax.bar_label(
        backbone_bars,
        labels=[f"{count:,}" for count in backbone_counts],
        padding=8,
        fontsize=7,
        rotation=25,
    )
    fig.tight_layout()

    saved = safe_savefig(fig, output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    if saved:
        print(f"Saved: {output_path}")


def main():
    """Compute and plot parameter counts for all experiment model architectures.

    Instantiates each architecture listed in :data:`MODEL_NAMES` using the
    specified ``--channel``, ``--num_classes``, and ``--input_size``, prints a
    summary line per model to stdout, and saves a grouped bar-chart PNG to
    ``--output``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--input_size", type=int, nargs=2, default=(32, 32), metavar=("H", "W"))
    parser.add_argument("--output", default="results/model_parameter_counts_log.png")
    args = parser.parse_args()

    counts = get_parameter_counts(args.channel, args.num_classes, tuple(args.input_size))
    for label, total, backbone in counts:
        print(f"{label}: {total:,} total; {backbone:,} without classifier or fully connected layers")
    save_parameter_count_plot(
        counts,
        args.output,
    )


if __name__ == "__main__":
    main()
