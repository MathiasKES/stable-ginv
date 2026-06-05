"""
Plot boxplots from masked registry keys using per-sample paired metrics.

Each masked key is looked up in masked_registry[_v2].json. The matching iDLG
baseline is found from the masked run's non-mask arguments, then paired sample
differences are computed on overlapping run_id ranges.

Examples:
    python helper/plot_registry_key_boxplots.py \
        --keys 47bd3143867404e817b709b8b7cefb8b d8287ce062fc58dfb17144d05bb07eea \
        --labels features.0 features.2 \
        --metric psnr \
        --out_dir results/ablation_boxplots \
        --output_prefix vgg13_lbfgs_no_pretrain

    python helper/plot_registry_key_boxplots.py \
        --spec_csv results/vgg13_ablation_keys.csv \
        --metric psnr \
        --out_dir results/ablation_boxplots

The optional spec CSV can use these columns:
    masked_key,label,group

It also accepts experiment-result CSVs where the registry key is stored as an
extra trailing column after png_path. If label/group are omitted, labels are
inferred from the CSV row or registry arguments.
"""
import argparse
import csv
import hashlib
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from functions.io_utils import find_registry_entry, resolve_storage_paths, safe_makedirs, safe_savefig


METRICS = {
    "psnr": ("best_psnr_list", "PSNR difference (masked - baseline, dB)"),
    "mse": ("best_mse_list", "MSE difference (masked - baseline)"),
    "ssim": ("best_ssim_list", "SSIM difference (masked - baseline)"),
}

ABLATION_LAYERS = {
    "LeNet": ["body.0", "body.2", "body.4", "fc"],
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

BASELINE_ARG_KEYS = [
    "dataset",
    "network",
    "pretrained",
    "lr",
    "gamma",
    "grad_loss",
    "num_dummy",
    "iteration",
    "tv_weight",
    "optimizer",
    "num_restarts",
    "max_iteration",
    "history_size",
]


def _load_registry(path):
    with open(path) as f:
        return json.load(f)


def _load_registry_with_legacy(path):
    legacy_path = path.replace("_registry_v2.json", "_registry.json")
    registry = {}
    loaded = []
    if os.path.isfile(legacy_path):
        registry.update(_load_registry(legacy_path))
        loaded.append(legacy_path)
    if os.path.isfile(path):
        registry.update(_load_registry(path))
        loaded.append(path)
    if not loaded:
        raise FileNotFoundError(f"No registry files found. Checked: {legacy_path}, {path}")
    print(f"Loaded: {', '.join(loaded)}")
    return registry


def _default_registry_paths():
    _, save_path = resolve_storage_paths(".")
    baseline_dir = os.path.join(save_path, "baselines")
    return (
        os.path.join(baseline_dir, "masked_registry_v2.json"),
        os.path.join(baseline_dir, "idlg_baselines_registry_v2.json"),
    )


def _registry_paths_from_baseline_dir(baseline_dir):
    return (
        os.path.join(baseline_dir, "masked_registry_v2.json"),
        os.path.join(baseline_dir, "idlg_baselines_registry_v2.json"),
    )


def _baseline_key_from_masked_args(masked_args):
    baseline_args = {
        key: masked_args[key]
        for key in BASELINE_ARG_KEYS
        if key in masked_args
    }
    key_json = json.dumps(baseline_args, sort_keys=True)
    return hashlib.md5(key_json.encode("utf-8")).hexdigest()


def _baseline_args_from_masked_args(masked_args):
    return {
        key: masked_args[key]
        for key in BASELINE_ARG_KEYS + ["num_exp", "run_id"]
        if key in masked_args
    }


def _config_label(args):
    optimizer = args.get("optimizer", "")
    opt_label = {
        "lbfgs": "L-BFGS",
        "signed_adamw": "Signed AdamW",
    }.get(optimizer, optimizer)
    pretrain = "Pretrained" if args.get("pretrained") else "No pretrain"
    return f"{opt_label} / {pretrain}".strip(" /")


def _normalize_prefix(prefix):
    if prefix.startswith("classifier"):
        return "classifier"
    if prefix.startswith("fc"):
        return "fc"
    return prefix


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _config_label_from_row(row):
    optimizer = row.get("optimizer", "")
    opt_label = {
        "lbfgs": "L-BFGS",
        "signed_adamw": "Signed AdamW",
    }.get(optimizer, optimizer)
    pretrain = "Pretrained" if _parse_bool(row.get("pretrained", False)) else "No pretrain"
    return f"{opt_label} / {pretrain}".strip(" /")


def _infer_ablation_label_from_row(row):
    explicit = row.get("label") or row.get("layer")
    if explicit:
        return explicit.strip()

    network = row.get("network", "")
    expected = ABLATION_LAYERS.get(network)
    prefixes = row.get("prefixes", "")
    if not expected or not prefixes:
        return ""

    kept = set()
    for part in prefixes.split(","):
        part = part.strip()
        if not part:
            continue
        prefix = _normalize_prefix(part.split(":", 1)[0].strip())
        if ":" in part:
            try:
                if float(part.split(":", 1)[1]) == 0:
                    continue
            except ValueError:
                pass
        kept.add(prefix)

    return _missing_layer_label(expected, kept)


def _missing_layer_label(expected, kept):
    missing = [layer for layer in expected if layer not in kept]
    non_classifier_missing = [layer for layer in missing if layer not in {"classifier", "fc"}]
    if len(non_classifier_missing) == 1:
        return non_classifier_missing[0]
    if len(missing) == 1:
        return missing[0]
    return ""


def _key_from_spec_row(row):
    for column in ("masked_key", "key", "registry_key"):
        value = row.get(column)
        if value:
            return value.strip()

    overflow = row.get(None) or []
    if overflow:
        return str(overflow[-1]).strip()
    return ""


def _infer_label(masked_args):
    prefixes = str(masked_args.get("prefixes", ""))
    if prefixes:
        kept = []
        for part in prefixes.split(","):
            part = part.strip()
            if not part:
                continue
            prefix = _normalize_prefix(part.split(":", 1)[0].strip())
            frac = None
            if ":" in part:
                try:
                    frac = float(part.split(":", 1)[1])
                except ValueError:
                    frac = None
            if frac != 0:
                kept.append(prefix)
        if kept:
            expected = ABLATION_LAYERS.get(masked_args.get("network", ""))
            if expected:
                label = _missing_layer_label(expected, set(kept))
                if label:
                    return label
            return "+".join(kept)
    return masked_args.get("mask_mode") or "masked"


def _parse_specs(args):
    specs = []
    if args.spec_csv:
        with open(args.spec_csv, newline="") as f:
            for row in csv.DictReader(f):
                method = (row.get("method") or "").strip().lower()
                if method and "masked" not in method:
                    continue
                key = _key_from_spec_row(row)
                if not key:
                    raise ValueError(
                        "spec_csv must contain masked_key/key/registry_key, "
                        "or a trailing registry key column"
                    )
                specs.append({
                    "masked_key": key,
                    "label": _infer_ablation_label_from_row(row),
                    "group": (
                        row.get("group")
                        or row.get("config")
                        or _config_label_from_row(row)
                    ).strip(),
                })

    for i, key in enumerate(args.keys or []):
        label = args.labels[i] if args.labels and i < len(args.labels) else ""
        group = args.groups[i] if args.groups and i < len(args.groups) else ""
        specs.append({"masked_key": key, "label": label, "group": group})

    if not specs:
        raise ValueError("Provide --keys or --spec_csv")
    return specs


def _metric_values(entry, metric):
    list_name, _ = METRICS[metric]
    return [float(v) for v in entry.get(list_name, [])]


def _paired_rows_for_spec(spec, masked_registry, baseline_registry, metric):
    masked_key = spec["masked_key"]
    masked_entry = masked_registry.get(masked_key)
    if masked_entry is None:
        if masked_key in baseline_registry:
            raise ValueError(
                f"Skipping baseline key from CSV, not a masked key: {masked_key}"
            )
        raise ValueError(f"Masked key not found: {masked_key}")

    masked_args = masked_entry.get("args", {})
    baseline_key = _baseline_key_from_masked_args(masked_args)
    baseline_args = _baseline_args_from_masked_args(masked_args)
    stored_baseline_key, baseline_entry = find_registry_entry(
        baseline_registry, baseline_key, baseline_args
    )
    if baseline_entry is None:
        raise ValueError(f"No matching baseline found for masked key: {masked_key}")

    masked_values = _metric_values(masked_entry, metric)
    baseline_values = _metric_values(baseline_entry, metric)
    if not masked_values:
        raise ValueError(f"Masked key has no {metric} values: {masked_key}")
    if not baseline_values:
        raise ValueError(f"Baseline key has no {metric} values: {stored_baseline_key}")

    masked_start = int(masked_args.get("run_id", 0))
    baseline_start = int(baseline_entry.get("args", {}).get("run_id", 0))
    overlap_start = max(masked_start, baseline_start)
    overlap_end = min(masked_start + len(masked_values), baseline_start + len(baseline_values))
    if overlap_start >= overlap_end:
        raise ValueError(
            f"No overlapping run_id range for masked={masked_key}, baseline={stored_baseline_key}"
        )

    label = spec.get("label") or _infer_label(masked_args)
    group = spec.get("group") or _config_label(masked_args)

    rows = []
    for run_id in range(overlap_start, overlap_end):
        masked_index = run_id - masked_start
        baseline_index = run_id - baseline_start
        masked_value = masked_values[masked_index]
        baseline_value = baseline_values[baseline_index]
        rows.append({
            "masked_key": masked_key,
            "baseline_key": stored_baseline_key,
            "label": label,
            "group": group,
            "run_id": run_id,
            "masked_index": masked_index,
            "baseline_index": baseline_index,
            "network": masked_args.get("network", ""),
            "dataset": masked_args.get("dataset", ""),
            "mask_mode": masked_args.get("mask_mode", ""),
            "prefixes": masked_args.get("prefixes", ""),
            f"masked_{metric}": masked_value,
            f"baseline_{metric}": baseline_value,
            f"delta_{metric}": masked_value - baseline_value,
        })
    return rows


def _build_rows(specs, masked_registry, baseline_registry, metric):
    rows = []
    for spec in specs:
        try:
            rows.extend(_paired_rows_for_spec(spec, masked_registry, baseline_registry, metric))
        except ValueError as exc:
            print(f"WARNING: {exc}")
    if not rows:
        raise ValueError("No paired rows could be built from the provided keys.")
    return pd.DataFrame(rows)


def _summary(df, metric):
    value_col = f"delta_{metric}"
    return (
        df.groupby(["group", "label", "masked_key", "baseline_key"], as_index=False)[value_col]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )


def _plot(df, metric, out_path, title=None):
    _, y_label = METRICS[metric]
    value_col = f"delta_{metric}"
    label_order = list(dict.fromkeys(df["label"]))
    group_count = df["group"].nunique()

    sns.set_theme(style="whitegrid", context="paper")
    if group_count > 1:
        height = max(3.0, 0.45 * len(label_order) + 1.5)
        grid = sns.catplot(
            data=df,
            kind="box",
            y="label",
            x=value_col,
            col="group",
            col_wrap=2,
            order=list(reversed(label_order)),
            sharex=True,
            sharey=True,
            height=height,
            aspect=1.25,
            color=sns.color_palette()[0],
            fliersize=0,
        )
        grid.map_dataframe(
            sns.stripplot,
            y="label",
            x=value_col,
            order=list(reversed(label_order)),
            color="0.2",
            size=2.2,
            alpha=0.55,
            jitter=0.18,
        )
        for ax in grid.axes.flat:
            ax.axvline(0, linestyle="--", linewidth=1, color="0.25")
            ax.set_xlabel(y_label)
            ax.set_ylabel("")
        grid.set_titles("{col_name}")
        if title:
            grid.figure.suptitle(title, y=1.03)
        grid.figure.tight_layout()
        fig = grid.figure
    else:
        fig, ax = plt.subplots(figsize=(8, max(3.0, 0.45 * len(label_order) + 1.5)))
        sns.boxplot(
            data=df,
            y="label",
            x=value_col,
            order=list(reversed(label_order)),
            color=sns.color_palette()[0],
            fliersize=0,
            ax=ax,
        )
        sns.stripplot(
            data=df,
            y="label",
            x=value_col,
            order=list(reversed(label_order)),
            color="0.2",
            size=2.2,
            alpha=0.55,
            jitter=0.18,
            ax=ax,
        )
        ax.axvline(0, linestyle="--", linewidth=1, color="0.25")
        ax.set_xlabel(y_label)
        ax.set_ylabel("")
        if title:
            ax.set_title(title)
        fig.tight_layout()

    safe_savefig(fig, out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    default_masked, default_baseline = _default_registry_paths()
    parser = argparse.ArgumentParser(
        description="Create paired metric-difference boxplots from masked registry keys."
    )
    parser.add_argument("--keys", nargs="*", help="Masked registry keys to plot.")
    parser.add_argument("--labels", nargs="*", help="Optional labels matching --keys.")
    parser.add_argument("--groups", nargs="*", help="Optional facet groups matching --keys.")
    parser.add_argument("--spec_csv", help="CSV with masked_key,label,group columns.")
    parser.add_argument("--metric", choices=METRICS, default="psnr")
    parser.add_argument(
        "--baseline_dir",
        help="Directory containing masked_registry*.json and idlg_baselines_registry*.json.",
    )
    parser.add_argument("--masked_registry_path", default=default_masked)
    parser.add_argument("--baseline_registry_path", default=default_baseline)
    parser.add_argument("--out_dir", default=os.path.join("results", "registry_key_boxplots"))
    parser.add_argument("--output_prefix", default="registry_key_boxplot")
    parser.add_argument("--title", default="")
    args = parser.parse_args()

    if args.labels and args.keys and len(args.labels) != len(args.keys):
        raise ValueError("--labels must have the same length as --keys")
    if args.groups and args.keys and len(args.groups) != len(args.keys):
        raise ValueError("--groups must have the same length as --keys")

    if args.baseline_dir:
        args.masked_registry_path, args.baseline_registry_path = _registry_paths_from_baseline_dir(
            args.baseline_dir
        )

    specs = _parse_specs(args)
    masked_registry = _load_registry_with_legacy(args.masked_registry_path)
    baseline_registry = _load_registry_with_legacy(args.baseline_registry_path)
    df = _build_rows(specs, masked_registry, baseline_registry, args.metric)

    safe_makedirs(args.out_dir)
    paired_csv = os.path.join(args.out_dir, f"{args.output_prefix}_paired_{args.metric}.csv")
    summary_csv = os.path.join(args.out_dir, f"{args.output_prefix}_summary_{args.metric}.csv")
    png_path = os.path.join(args.out_dir, f"{args.output_prefix}_{args.metric}_boxplot.png")
    pdf_path = os.path.join(args.out_dir, f"{args.output_prefix}_{args.metric}_boxplot.pdf")

    df.to_csv(paired_csv, index=False)
    _summary(df, args.metric).to_csv(summary_csv, index=False)
    print(f"Saved: {paired_csv}")
    print(f"Saved: {summary_csv}")

    title = args.title or None
    _plot(df, args.metric, png_path, title=title)
    _plot(df, args.metric, pdf_path, title=title)


if __name__ == "__main__":
    main()
