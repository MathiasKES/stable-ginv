"""CLI for the Jacobian rank-vs-gradient-budget sweep (multiprocessing + plotting)."""
import os
import csv
import argparse
from datetime import datetime

import numpy as np
import torch.multiprocessing as mp

from stable_ginv.jacobian.sweep import _run_dtype, _seed_all

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from stable_ginv.io import (
    parse_prefixes_with_fracs,
    resolve_storage_paths,
    safe_makedirs,
    safe_savefig,
)
from functions.Dataset import load_dataset

sns.set_theme(style="whitegrid")


def _parse_explicit_sample_indices(sample_indices_str, dataset_size):
    """Parse a '--sample_indices' string into a validated list of dataset indices.

    Raises ValueError if any index falls outside [0, dataset_size).
    """
    indices = [int(x.strip()) for x in sample_indices_str.split(",") if x.strip()]
    if any(i < 0 or i >= dataset_size for i in indices):
        raise ValueError(
            f"--sample_indices contains an index out of range for dataset of size {dataset_size}"
        )
    return indices


def main():
    """Parse CLI arguments and run the Jacobian rank sweep.

    Builds the gradient-budget grid (``--row_counts`` or ``--stepsize``),
    dispatches per-dtype sweeps via :func:`~stable_ginv.jacobian.sweep._run_dtype`
    (serial or multiprocessing), then writes the summary CSV and a rank-vs-rows
    line plot to the configured save directory.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--network", type=str, default="resnet18")
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--method", type=str, default="idlg", choices=["idlg", "masked"])
    parser.add_argument("--pretrained", action="store_true",
                        help="Load ImageNet-pretrained weights (affects normalization).")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"],
                        help="Floating-point dtype used for the network, input, gradients, "
                             "and Jacobian construction. Default preserves previous behavior.")
    parser.add_argument("--both_dtypes", action="store_true",
                        help="Run the same sweep twice, once with float32 and once with float64, "
                             "and plot both rank curves in one graph.")

    parser.add_argument(
        "--mask_mode",
        type=str,
        default="gradsize_topfrac",
        choices=[
            "none",
            "all",
            "gradsize_topk",
            "gradsize_topfrac",
            "gradsize_topk_entries",
            "gradsize_topfrac_entries",
            "gradsize_topk_entries_layer",
            "gradsize_topfrac_entries_layer",
            "prefix",
            "prefix_topk",
            "prefix_topfrac",
            "prefix_topk_entries",
            "prefix_topfrac_entries",
            "prefix_topk_entries_layer",
            "prefix_topfrac_entries_layer",
        ],
    )
    parser.add_argument("--prefixes", type=str, default="conv1:1.0,layer1:1.0,layer2:1.0,layer3:1.0,fc:1.0")
    parser.add_argument("--gradsize_topk", type=int, default=20)
    parser.add_argument("--gradsize_topfrac", type=float, default=0.5)

    parser.add_argument("--row_counts", type=str,
                        default="3072,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000")
    parser.add_argument("--stepsize", type=int, default=None,
                        help="Step between row counts. Generates range from --min_row_count "
                             "(default: unknowns) to --max_row_count in increments of stepsize. "
                             "Overrides --row_counts when set.")
    parser.add_argument("--max_row_count", type=int, default=None,
                        help="Upper bound for row counts when --stepsize is used.")
    parser.add_argument("--min_row_count", type=int, default=None,
                        help="Smallest row count to sweep when --stepsize is used. "
                             "Defaults to unknowns (= C×H×W).")
    parser.add_argument("--jacobian_select_mode", type=str, default="topk_abs",
                        choices=["topk_abs", "first", "random", "layer_spread"])
    parser.add_argument("--qr_pivot", action="store_true",
                        help="After building J with --jacobian_select_mode, reorder its rows "
                             "via QR column pivoting so each slice J[:k] contains the k most "
                             "linearly independent rows from the pool.")
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--sample_indices", type=str, default=None,
                        help="Comma-separated dataset indices to use instead of random sampling, "
                             "e.g. --sample_indices 39508,23784. Overrides --num_samples and --run_id.")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of parallel workers (1 = serial, 2-4 = one process per GPU).")
    parser.add_argument("--print_svd_info", action="store_true",
                        help="Print sigma_max, atol, and the 10 smallest singular values "
                             "of the Jacobian at each row count.")
    parser.add_argument("--normalise", action="store_true",
                        help="L2-normalise rows of J before computing matrix_rank. "
                             "Default is no normalisation: the relative threshold reflects "
                             "actual gradient magnitudes rather than directions.")
    parser.add_argument("--no_independent", action="store_true",
                        help="Use pool-slice mode: build J once at max(row_counts) and slice "
                             "J[:k] for each k. Faster but rank at k depends on the pool composition. "
                             "Default is independent mode (build J separately for each k).")
    parser.add_argument("--exclude_fc", action="store_true",
                        help="Exclude the last FC layer from the Jacobian entirely — "
                             "not in the pool, not forced. FC entries are removed from keep_ids "
                             "before the sweep runs.")

    args = parser.parse_args()

    if not 1 <= args.num_workers <= 4:
        parser.error("--num_workers must be between 1 and 4")
    if not 0 < args.gradsize_topfrac <= 1:
        parser.error("--gradsize_topfrac must be in (0, 1]")
    if args.num_samples < 1:
        parser.error("--num_samples must be at least 1")

    if args.stepsize is not None:
        if args.max_row_count is None:
            parser.error("--max_row_count is required when --stepsize is used")
        if args.stepsize <= 0:
            parser.error("--stepsize must be a positive integer")
        if args.max_row_count <= 0:
            parser.error("--max_row_count must be a positive integer")
        if args.min_row_count is not None and args.min_row_count <= 0:
            parser.error("--min_row_count must be a positive integer")
        row_counts = None  # built after dataset load
    else:
        row_counts = [int(x.strip()) for x in args.row_counts.split(",") if x.strip()]
        if len(row_counts) == 0:
            parser.error("--row_counts must contain at least one positive integer")
        if any(rows <= 0 for rows in row_counts):
            parser.error("--row_counts values must all be positive")
    prefixes, prefix_layer_fracs = parse_prefixes_with_fracs(args.prefixes)
    if any(not 0 < frac <= 1 for frac in prefix_layer_fracs.values()):
        parser.error("prefix fractions in --prefixes must be in (0, 1]")

    data_path, save_dir = resolve_storage_paths()
    safe_makedirs(save_dir)

    dst, channel, num_classes, shape_img = load_dataset(args.dataset, data_path)

    if args.sample_indices is not None:
        try:
            sample_indices = _parse_explicit_sample_indices(args.sample_indices, len(dst))
        except ValueError as exc:
            parser.error(str(exc))
    else:
        seed = args.run_id + 1
        _seed_all(seed)
        idx_shuffle = np.random.permutation(len(dst))
        sample_indices = idx_shuffle[:args.num_samples].tolist()
    sample_count = len(sample_indices)

    unknowns = channel * shape_img[0] * shape_img[1]

    if args.stepsize is not None:
        min_row_count = args.min_row_count if args.min_row_count is not None else (unknowns // 1000) * 1000
        if min_row_count > args.max_row_count:
            parser.error(f"--min_row_count ({min_row_count}) exceeds --max_row_count ({args.max_row_count})")
        row_counts = list(range(min_row_count, args.max_row_count + 1, args.stepsize))
        if not row_counts or row_counts[-1] < args.max_row_count:
            row_counts.append(args.max_row_count)

    if unknowns not in row_counts:
        row_counts = sorted(row_counts + [unknowns])

    print(f"Unknowns: {unknowns}")
    print(f"Samples: {sample_indices}")
    dtype_values = ["float32", "float64"] if args.both_dtypes else [args.dtype]
    print(f"Dtype(s): {dtype_values}")

    # AD passes per sample: fwAD when unknowns < k (unknowns passes), else backward (k passes).
    rank_steps = len(row_counts) * (2 if args.qr_pivot else 1)
    if not args.no_independent:
        j_steps = sum(min(k, unknowns) for k in row_counts)
    else:
        j_steps = unknowns  # one build at max(row_counts), always fwAD when max > unknowns
    total_steps = len(sample_indices) * (j_steps + rank_steps)

    runs = {}
    for dtype_name in dtype_values:
        runs[dtype_name] = _run_dtype(
            args, dtype_name, sample_indices, row_counts, prefixes, prefix_layer_fracs, total_steps
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dtype_label = "both_dtypes" if args.both_dtypes else dtype_values[0]
    base = (
        f"jac_rank_{args.network}_{args.dataset}_{args.method}_"
        f"{args.jacobian_select_mode}_{dtype_label}_ns{sample_count}_{timestamp}"
    )

    csv_path = os.path.join(save_dir, base + ".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["rows_used", "mean_rank", "std_rank",
                  "unknowns", "num_samples", "jacobian_select_mode", "dtype", "independent",
                  "sample_indices", "per_sample_ranks"]
        if args.qr_pivot:
            header += ["mean_rank_qr", "std_rank_qr", "per_sample_ranks_qr"]
        writer.writerow(header)
        sample_indices_str = ";".join(str(i) for i in sample_indices)
        for dtype_name, run in runs.items():
            rank_results = run["rank_results"]
            rank_results_qr = run["rank_results_qr"]
            for i, (x, m, s) in enumerate(zip(run["xs"], run["mean_ranks"], run["std_ranks"])):
                ranks_str = ";".join(str(r) for r in rank_results[x])
                row = [x, m, s, unknowns, sample_count, args.jacobian_select_mode, dtype_name, not args.no_independent,
                       sample_indices_str, ranks_str]
                if rank_results_qr is not None:
                    ranks_qr_str = ";".join(str(r) for r in rank_results_qr[x])
                    row += [run["mean_ranks_qr"][i], run["std_ranks_qr"][i], ranks_qr_str]
                writer.writerow(row)

    plot_rows = []
    for dtype_name, run in runs.items():
        legend_label = f"{dtype_name} (select={args.jacobian_select_mode}, samples={sample_count}"
        if args.method == "masked":
            legend_label += f", mask={args.mask_mode}"
        legend_label += ")"
        for rows_used, ranks in run["rank_results"].items():
            for rank in ranks:
                plot_rows.append({
                    "Rows used": rows_used,
                    "Jacobian rank": rank,
                    "Series": legend_label,
                })
        if run["rank_results_qr"] is not None:
            qr_label = f"{dtype_name} qr_pivot ({args.jacobian_select_mode} pool)"
            for rows_used, ranks in run["rank_results_qr"].items():
                for rank in ranks:
                    plot_rows.append({
                        "Rows used": rows_used,
                        "Jacobian rank": rank,
                        "Series": qr_label,
                    })

    import pandas as pd
    fig, ax = plt.subplots(figsize=(7, 5))
    if plot_rows:
        sns.lineplot(
            data=pd.DataFrame(plot_rows),
            x="Rows used",
            y="Jacobian rank",
            hue="Series",
            style="Series",
            markers=True,
            dashes=True,
            errorbar="sd",
            ax=ax,
        )
    ax.axhline(unknowns, linestyle=":", label=f"unknowns = {unknowns}")
    ax.set_xlabel("Number of Jacobian rows / gradients used")
    ax.set_ylabel("Average Jacobian rank")
    ax.set_title(f"Jacobian rank sweep: {args.network}, {args.dataset}, {args.method}")
    ax.legend()
    fig.tight_layout()

    plot_path = os.path.join(save_dir, base + ".png")
    safe_savefig(fig, plot_path, dpi=200)
    plt.close(fig)

    print(f"\nSaved CSV to: {csv_path}")
    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
