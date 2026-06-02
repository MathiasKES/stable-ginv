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
1a) From a JSON file that references registry uids (recommended):

       python manual_stats.py data.json

   where data.json looks like:

       {
         "baseline": "2bbb422a71a492f69a1cf989866331b8",
         "masked":   "63b40d0329a5b507a1cc72ee33bd169c"
       }

   "masked" may also be a list of uids, each compared against the baseline:

       {
         "baseline": "2bbb422a71a492f69a1cf989866331b8",
         "masked": ["63b40d03...", "0f99ecb1...", "f5f0b18b..."]
       }

   The baseline uid is looked up in idlg_baselines_registry.json and each
   masked uid in masked_registry.json (default locations come from
   resolve_storage_paths). The best_mse_list / best_psnr_list / best_ssim_list
   stored for each uid are used directly. Override registry locations with
   optional "idlg_registry" / "masked_registry" keys in data.json.

1b) From a JSON file with inline metric lists:

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

Output:
    A JSON object keyed by registry uid is written/merged, e.g.

        {
          "<baseline_uid>": {"method": "idlg",   ..., "ssim_normality": ""},
          "<masked_uid>":   {"method": "masked", ..., "ssim_normality": "..."}
        }

    Each run adds/updates the baseline (method="idlg") and masked
    (method="masked") uid entries. The baseline entry carries the aggregate
    columns (method .. std_best_ssim); the <measure>_{ci,significant,normality}
    columns are populated only on the masked entry. nan values are written as
    "". Default output path:
        /work3/s234843/bachelor/results/baselines/exp_results_resnet_manual.json

Flags:
    --psnr-from-mse   Ignore any inserted PSNR and recompute it from MSE using
                      the pipeline's compute_psnr_from_mse(mse, max_val=1.0).
    --out=PATH.json   Append entries to PATH.json instead of the default file.
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
from functions.io_utils import (  # noqa: E402
    resolve_storage_paths,
    safe_chmod,
    safe_makedirs,
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

# Inner-entry header columns (the registry uid is the top-level JSON key, not a field).
HEADER_COLUMNS = ["method", "network", "grad_loss", "pretrained", "prefixes"]
ENTRY_KEYS = HEADER_COLUMNS + STAT_COLUMNS
# Paired-comparison columns: only populated for the masked entry.
CI_KEYS = {
    "mse_ci", "mse_significant", "psnr_ci", "psnr_significant", "psnr_normality",
    "mse_normality", "ssim_ci", "ssim_significant", "ssim_normality",
}

DEFAULT_OUT = "/work3/s234843/bachelor/results/baselines/exp_results_resnet_manual.json"


def _blank(value):
    """Render nan / None as an empty string; leave everything else as-is."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def build_entry(meta_side, method, stat_row, include_ci):
    """Build one ordered output entry (the inner value for a uid key).

    nan/None numeric values become "". For the baseline (include_ci=False) the
    paired-comparison columns are left empty. The registry uid is the dict key,
    so it is not repeated inside the entry.
    """
    entry = {
        "method": method,
        "network": meta_side["network"],
        "grad_loss": meta_side["grad_loss"],
        "pretrained": meta_side["pretrained"],
        "prefixes": meta_side["prefixes"],
    }
    for key in STAT_COLUMNS:
        if key in CI_KEYS:
            entry[key] = _blank(stat_row.get(key, "")) if include_ci else ""
        else:
            entry[key] = _blank(stat_row.get(key))
    return entry


def build_baseline_entry(baseline_meta, stats):
    """Return (uid_key, entry) for the iDLG baseline."""
    entry = build_entry(baseline_meta, "idlg", baseline_stat_row(stats), include_ci=False)
    return baseline_meta["registry"] or "idlg", entry


def build_masked_entry(masked_meta, stats, paired_report):
    """Return (uid_key, entry) for one masked comparison against the baseline."""
    entry = build_entry(masked_meta, "masked", masked_stat_row(stats, paired_report), include_ci=True)
    return masked_meta["registry"] or "masked", entry


def append_entries_json(path, entries):
    """Merge uid->entry mappings into a JSON object file, creating it if needed."""
    existing = {}
    if os.path.isfile(path):
        with open(path) as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = {}
        if not isinstance(existing, dict):
            raise SystemExit(
                f"{path} is not a JSON object keyed by uid; refusing to overwrite."
            )
    existing.update(entries)

    safe_makedirs(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
    safe_chmod(path)
    print(f"\nWrote {len(entries)} uid entr(y/ies) to {path} (total {len(existing)}).")


def print_report(stats, paired_report, baseline_label="", masked_label=""):
    base = baseline_stat_row(stats)
    mask = masked_stat_row(stats, paired_report)

    print(f"\n=== iDLG baseline aggregates {('['+baseline_label+']') if baseline_label else ''} ===")
    for k in STAT_COLUMNS:
        if k in base:
            print(f"  {k:18s}: {base[k]}")

    print(f"\n=== Masked aggregates + paired comparison (masked vs iDLG) "
          f"{('['+masked_label+']') if masked_label else ''} ===")
    for k in STAT_COLUMNS:
        print(f"  {k:18s}: {mask[k]}")


def _default_registry_paths():
    """Return (idlg_registry_path, masked_registry_path) the pipeline uses."""
    _, save_path = resolve_storage_paths(".")
    baseline_dir = os.path.join(save_path, "baselines")
    return (
        os.path.join(baseline_dir, "idlg_baselines_registry.json"),
        os.path.join(baseline_dir, "masked_registry.json"),
    )


def _entry_to_metrics(registry, uid, registry_path):
    """Extract {mse, psnr, ssim} lists for one uid from a loaded registry."""
    if uid not in registry:
        raise SystemExit(
            f"uid {uid!r} not found in registry {registry_path}. "
            f"Available uids: {', '.join(list(registry)[:5])}"
            + (" ..." if len(registry) > 5 else "")
        )
    entry = registry[uid]
    for key in ("best_mse_list", "best_psnr_list"):
        if key not in entry:
            raise SystemExit(f"uid {uid!r} in {registry_path} is missing '{key}'.")
    metrics = {
        "mse": entry["best_mse_list"],
        "psnr": entry["best_psnr_list"],
    }
    # SSIM may be absent in older baselines; omit so it is treated as missing.
    if entry.get("best_ssim_list") is not None:
        metrics["ssim"] = entry["best_ssim_list"]
    return metrics


def _empty_meta():
    return {"registry": "", "network": "", "grad_loss": "", "pretrained": "", "prefixes": ""}


def _meta_from_args(uid, args):
    """Pull the output header fields for one registry entry from its args block."""
    return {
        "registry": uid,
        "network": args.get("network", ""),
        "grad_loss": args.get("grad_loss", ""),
        "pretrained": args.get("pretrained", ""),
        "prefixes": args.get("prefixes", ""),
    }


def load_from_registries(baseline_uid, masked_uids, data):
    """Resolve baseline + one-or-more masked metric lists by uid from the registries.

    The baseline uid is looked up in idlg_baselines_registry.json and each
    masked uid in masked_registry.json. "masked" may be a single uid string or
    a list of uid strings (each compared against the baseline). Registry paths
    default to the pipeline locations (resolve_storage_paths) but may be
    overridden in data.json via "idlg_registry" / "masked_registry".

    Returns (baseline_metrics, baseline_meta, masked_items) where masked_items
    is a list of {"metrics": ..., "meta": ...} dicts.
    """
    if isinstance(masked_uids, str):
        masked_uids = [masked_uids]

    default_idlg, default_masked = _default_registry_paths()
    idlg_path = data.get("idlg_registry", default_idlg)
    masked_path = data.get("masked_registry", default_masked)

    with open(idlg_path) as f:
        idlg_registry = json.load(f)
    with open(masked_path) as f:
        masked_registry = json.load(f)

    baseline = _entry_to_metrics(idlg_registry, baseline_uid, idlg_path)
    baseline_meta = _meta_from_args(baseline_uid, idlg_registry[baseline_uid].get("args", {}))
    print(f"Resolved baseline uid {baseline_uid} from {idlg_path}")

    masked_items = []
    for muid in masked_uids:
        masked_items.append({
            "metrics": _entry_to_metrics(masked_registry, muid, masked_path),
            "meta": _meta_from_args(muid, masked_registry[muid].get("args", {})),
        })
        print(f"Resolved masked   uid {muid} from {masked_path}")
    return baseline, baseline_meta, masked_items


def load_json(path):
    with open(path) as f:
        data = json.load(f)
    baseline = data["baseline"]
    masked = data["masked"]
    # uid-lookup form: "baseline" is a registry uid string (masked: str or list).
    if isinstance(baseline, str):
        return load_from_registries(baseline, masked, data)
    # inline form: single baseline/masked metric dicts.
    return baseline, _empty_meta(), [{"metrics": masked, "meta": _empty_meta()}]


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
    return baseline, _empty_meta(), [{"metrics": masked, "meta": _empty_meta()}]


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
    out_path = DEFAULT_OUT
    for a in argv[1:]:
        if a.startswith("--out="):
            out_path = a.split("=", 1)[1]

    if args:
        path = args[0]
        if path.lower().endswith(".json"):
            baseline, baseline_meta, masked_items = load_json(path)
        elif path.lower().endswith(".csv"):
            baseline, baseline_meta, masked_items = load_csv(path)
        else:
            raise SystemExit(f"Unrecognised input extension: {path} (use .json or .csv)")
        print(f"Loaded {len(baseline['mse'])} experiment(s) and "
              f"{len(masked_items)} masked comparison(s) from {path}")
    else:
        baseline = EXAMPLE["baseline"]
        baseline_meta = _empty_meta()
        masked_items = [{"metrics": EXAMPLE["masked"], "meta": _empty_meta()}]
        print(f"No input file given; using EXAMPLE ({len(baseline['mse'])} experiments). "
              "Edit the EXAMPLE dict or pass a .json/.csv file.")

    entries = {}
    baseline_done = False
    for item in masked_items:
        results = build_results(baseline, item["metrics"], psnr_from_mse=psnr_from_mse)
        stats, paired_report = compute(results)
        print_report(
            stats, paired_report,
            baseline_label=baseline_meta["registry"],
            masked_label=item["meta"]["registry"],
        )
        # Baseline aggregates are identical across masked comparisons; emit once.
        if not baseline_done:
            base_key, base_entry = build_baseline_entry(baseline_meta, stats)
            entries[base_key] = base_entry
            baseline_done = True
        mask_key, mask_entry = build_masked_entry(item["meta"], stats, paired_report)
        entries[mask_key] = mask_entry

    append_entries_json(out_path, entries)


if __name__ == "__main__":
    main(sys.argv)
