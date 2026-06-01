# manual_stats.py
"""
Manually insert per-experiment baseline (iDLG) and masked best MSE / PSNR / SSIM
values and compute the exact same aggregate + paired statistics that
iDLG_mask.py writes to the experiment CSV.

The statistics are NOT re-implemented here: this script feeds the inserted
values through the same functions the real pipeline uses
(functions.experiment_results + functions.io_utils), so every column matches
exactly how it would have been produced by a real run:

    med_best_loss, avg_best_loss, med_best_mse, avg_best_mse,
    avg_best_psnr, std_best_psnr, avg_best_ssim, std_best_ssim,
    mse_ci, mse_significant, psnr_ci, psnr_significant, psnr_normality,
    mse_normality, ssim_ci, ssim_significant, ssim_normality

Per-experiment pairing is by list position: baseline[i] is paired with
masked[i] (one image / experiment per index), exactly as in
paired_report_for_both.

--------------------------------------------------------------------------
Usage
--------------------------------------------------------------------------
1) From a JSON file:

       python manual_stats.py data.json

   where data.json looks like:

       {
         "baseline": {
           "mse":  [0.012, 0.034, ...],
           "psnr": [19.2, 14.7, ...],          # optional (see --psnr-from-mse)
           "ssim": [0.81, 0.55, ...],          # optional
           "loss": [1.2e-4, 3.4e-4, ...]       # optional
         },
         "masked": {
           "mse":  [0.020, 0.041, ...],
           "psnr": [17.0, 13.9, ...],
           "ssim": [0.74, 0.50, ...],
           "loss": [2.1e-4, 5.0e-4, ...]
         }
       }

2) From a CSV file with one row per experiment and any of these columns:

       mse_idlg,psnr_idlg,ssim_idlg,loss_idlg,mse_masked,psnr_masked,ssim_masked,loss_masked

       python manual_stats.py data.csv

3) With no file: edit the EXAMPLE dict near the bottom of this file and run

       python manual_stats.py

Flags:
    --psnr-from-mse   Ignore any inserted PSNR and recompute it from MSE using
                      the pipeline's compute_psnr_from_mse(mse, max_val=1.0).
    --out PATH.csv    Also append the resulting masked-method stat row to a CSV.
"""
import csv
import json
import math
import os
import sys

# Make `functions` / `helper` importable regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from functions.experiment_results import (  # noqa: E402
    append_result_metrics,
    compute_aggregate_stats,
    create_metric_accumulators,
    paired_report_for_both,
)
from helper.metrics import compute_psnr_from_mse  # noqa: E402


def _as_list(d, key, n):
    """Return d[key] as a list of length n, or a list of None when absent."""
    vals = d.get(key)
    if vals is None:
        return [None] * n
    vals = list(vals)
    if len(vals) != n:
        raise ValueError(
            f"'{key}' has {len(vals)} values but expected {n} (must match the "
            f"number of experiments / the length of the MSE list)."
        )
    return vals


def build_results(baseline, masked, psnr_from_mse=False):
    """Construct per-experiment result dicts using the exact pipeline keys.

    Pairing is positional: result idx i holds baseline[i] and masked[i].
    """
    base_mse = list(baseline["mse"])
    mask_mse = list(masked["mse"])
    n = len(base_mse)
    if len(mask_mse) != n:
        raise ValueError(
            f"baseline mse has {n} values but masked mse has {len(mask_mse)}; "
            "paired statistics require one masked value per baseline value."
        )

    base_psnr = _as_list(baseline, "psnr", n)
    base_ssim = _as_list(baseline, "ssim", n)
    base_loss = _as_list(baseline, "loss", n)
    mask_psnr = _as_list(masked, "psnr", n)
    mask_ssim = _as_list(masked, "ssim", n)
    mask_loss = _as_list(masked, "loss", n)

    results = {}
    for i in range(n):
        b_psnr = base_psnr[i]
        m_psnr = mask_psnr[i]
        if psnr_from_mse or b_psnr is None:
            b_psnr = compute_psnr_from_mse(base_mse[i], max_val=1.0)
        if psnr_from_mse or m_psnr is None:
            m_psnr = compute_psnr_from_mse(mask_mse[i], max_val=1.0)

        results[i] = {
            # iDLG baseline (note the exact key casing used by the pipeline)
            "best_mse_iDLG": base_mse[i],
            "best_psnr_idlg": b_psnr,
            "best_ssim_idlg": base_ssim[i],
            "best_loss_iDLG": base_loss[i],
            # masked
            "best_mse_iDLG_masked": mask_mse[i],
            "best_psnr_masked": m_psnr,
            "best_ssim_masked": mask_ssim[i],
            "best_loss_iDLG_masked": mask_loss[i],
        }
    return results


def compute(results):
    """Run the inserted results through the real pipeline stat functions."""
    accumulators = create_metric_accumulators()
    for idx in sorted(results):
        append_result_metrics(results[idx], accumulators)

    stats = compute_aggregate_stats(accumulators)
    paired_report = paired_report_for_both(results, print)
    return stats, paired_report


def _round(value, ndigits):
    """round() that tolerates nan/inf, matching build_exp_result_rows usage."""
    try:
        return round(value, ndigits)
    except (TypeError, ValueError):
        return value


def masked_stat_row(stats, paired_report):
    """Build the masked-method stat columns exactly as build_exp_result_rows."""
    return {
        "med_best_loss": _round(stats["med_best_loss_masked"], 5),
        "avg_best_loss": _round(stats["avg_best_loss_masked"], 5),
        "med_best_mse": _round(stats["med_best_mse_masked"], 10),
        "avg_best_mse": _round(stats["avg_best_mse_masked"], 10),
        "avg_best_psnr": _round(stats["avg_best_psnr_masked"], 5),
        "std_best_psnr": _round(stats["std_best_psnr_masked"], 5),
        "avg_best_ssim": _round(stats["avg_best_ssim_masked"], 5),
        "std_best_ssim": _round(stats["std_best_ssim_masked"], 5),
        "mse_ci": paired_report["mse_ci_str"],
        "mse_significant": paired_report["mse_significant_str"],
        "psnr_ci": paired_report["psnr_ci_str"],
        "psnr_significant": paired_report["psnr_significant_str"],
        "psnr_normality": paired_report["psnr_normality_str"],
        "mse_normality": paired_report["mse_normality_str"],
        "ssim_ci": paired_report["ssim_ci_str"],
        "ssim_significant": paired_report["ssim_significant_str"],
        "ssim_normality": paired_report["ssim_normality_str"],
    }


def baseline_stat_row(stats):
    """Aggregate columns for the iDLG baseline method (no paired stats)."""
    return {
        "med_best_loss": _round(stats["med_best_loss_idlg"], 5),
        "avg_best_loss": _round(stats["avg_best_loss_idlg"], 5),
        "med_best_mse": _round(stats["med_best_mse_idlg"], 10),
        "avg_best_mse": _round(stats["avg_best_mse_idlg"], 10),
        "avg_best_psnr": _round(stats["avg_best_psnr_idlg"], 5),
        "std_best_psnr": _round(stats["std_best_psnr_idlg"], 5),
        "avg_best_ssim": _round(stats["avg_best_ssim_idlg"], 5),
        "std_best_ssim": _round(stats["std_best_ssim_idlg"], 5),
    }


STAT_COLUMNS = [
    "med_best_loss", "avg_best_loss", "med_best_mse", "avg_best_mse",
    "avg_best_psnr", "std_best_psnr", "avg_best_ssim", "std_best_ssim",
    "mse_ci", "mse_significant", "psnr_ci", "psnr_significant",
    "psnr_normality", "mse_normality", "ssim_ci", "ssim_significant",
    "ssim_normality",
]


def print_report(stats, paired_report):
    base = baseline_stat_row(stats)
    mask = masked_stat_row(stats, paired_report)

    print("\n=== iDLG baseline aggregates ===")
    for k in STAT_COLUMNS:
        if k in base:
            print(f"  {k:18s}: {base[k]}")

    print("\n=== Masked aggregates + paired comparison (masked vs iDLG) ===")
    for k in STAT_COLUMNS:
        print(f"  {k:18s}: {mask[k]}")

    print("\n=== Masked row, requested column order ===")
    print(",".join(STAT_COLUMNS))
    print(",".join(str(mask[k]) for k in STAT_COLUMNS))


def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data["baseline"], data["masked"]


def load_csv(path):
    baseline = {"mse": [], "psnr": [], "ssim": [], "loss": []}
    masked = {"mse": [], "psnr": [], "ssim": [], "loss": []}
    col_map = {
        "mse_idlg": (baseline, "mse"), "psnr_idlg": (baseline, "psnr"),
        "ssim_idlg": (baseline, "ssim"), "loss_idlg": (baseline, "loss"),
        "mse_masked": (masked, "mse"), "psnr_masked": (masked, "psnr"),
        "ssim_masked": (masked, "ssim"), "loss_masked": (masked, "loss"),
    }
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        present = [c for c in reader.fieldnames if c in col_map]
        for row in reader:
            for col in present:
                target, key = col_map[col]
                cell = row[col].strip() if row[col] is not None else ""
                target[key].append(float(cell) if cell != "" else None)

    # Drop fully-absent optional metric lists so they are treated as missing.
    for d in (baseline, masked):
        for key in ("psnr", "ssim", "loss"):
            if all(v is None for v in d[key]):
                d.pop(key)
    return baseline, masked


# --------------------------------------------------------------------------
# Edit this when running with no input file.
# --------------------------------------------------------------------------
EXAMPLE = {
    "baseline": {
        "mse":  [0.012, 0.034, 0.009, 0.051],
        "psnr": [19.21, 14.68, 20.46, 12.92],
        "ssim": [0.81, 0.55, 0.86, 0.49],
    },
    "masked": {
        "mse":  [0.020, 0.041, 0.015, 0.060],
        "psnr": [16.99, 13.87, 18.24, 12.22],
        "ssim": [0.74, 0.50, 0.79, 0.45],
    },
}


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    flags = {a for a in argv[1:] if a.startswith("--")}
    psnr_from_mse = "--psnr-from-mse" in flags
    out_path = None
    for a in argv[1:]:
        if a.startswith("--out="):
            out_path = a.split("=", 1)[1]

    if args:
        path = args[0]
        if path.lower().endswith(".json"):
            baseline, masked = load_json(path)
        elif path.lower().endswith(".csv"):
            baseline, masked = load_csv(path)
        else:
            raise SystemExit(f"Unrecognised input extension: {path} (use .json or .csv)")
        print(f"Loaded {len(baseline['mse'])} experiment(s) from {path}")
    else:
        baseline, masked = EXAMPLE["baseline"], EXAMPLE["masked"]
        print(f"No input file given; using EXAMPLE ({len(baseline['mse'])} experiments). "
              "Edit the EXAMPLE dict or pass a .json/.csv file.")

    results = build_results(baseline, masked, psnr_from_mse=psnr_from_mse)
    stats, paired_report = compute(results)
    print_report(stats, paired_report)

    if out_path:
        row = masked_stat_row(stats, paired_report)
        file_exists = os.path.isfile(out_path)
        with open(out_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=STAT_COLUMNS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        print(f"\nAppended masked stat row to {out_path}")


if __name__ == "__main__":
    main(sys.argv)
