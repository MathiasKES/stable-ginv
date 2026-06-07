"""Result aggregation, paired-report construction, and experiment-results CSV rows."""
import hashlib
import json
import os

import numpy as np

from stable_ginv.io import append_csv_row
from stable_ginv.stats import (
    mean_or_nan,
    median_or_nan,
    paired_metric_summaries,
    std_or_nan,
)


METRIC_ACCUMULATOR_KEYS = [
    ("best_ssim_idlg", "best_ssim_idlg"),
    ("best_ssim_masked", "best_ssim_masked"),
    ("last_psnr_idlg", "psnr_idlg"),
    ("last_psnr_masked", "psnr_masked"),
    ("last_loss_iDLG", "final_loss_idlg"),
    ("last_mse_iDLG", "final_mse_idlg"),
    ("last_loss_iDLG_masked", "final_loss_masked"),
    ("last_mse_iDLG_masked", "final_mse_masked"),
    ("best_psnr_idlg", "best_psnr_idlg"),
    ("best_psnr_masked", "best_psnr_masked"),
    ("best_loss_iDLG", "best_loss_idlg"),
    ("best_mse_iDLG", "best_mse_idlg"),
    ("best_loss_iDLG_masked", "best_loss_masked"),
    ("best_mse_iDLG_masked", "best_mse_masked"),
    ("psnr_per_restart_idlg", "psnr_per_restart_idlg"),
    ("psnr_per_restart_masked", "psnr_per_restart_masked"),
    ("mse_per_restart_idlg", "mse_per_restart_idlg"),
    ("mse_per_restart_masked", "mse_per_restart_masked"),
]


EMPTY_PAIRED_STATS = {
    "n": float("nan"),
    "mean_diff": float("nan"),
    "std_diff": float("nan"),
    "ci_low": float("nan"),
    "ci_high": float("nan"),
    "t_stat": float("nan"),
    "p_value": float("nan"),
}


EXP_RESULT_EXTRA_FIELDS = [
    "mask_mode",
    "prefixes",
    "grad_param",
    "med_best_loss",
    "avg_best_loss",
    "med_best_mse",
    "avg_best_mse",
    "avg_best_psnr",
    "std_best_psnr",
    "avg_best_ssim",
    "std_best_ssim",
    "mse_ci",
    "mse_significant",
    "psnr_ci",
    "psnr_significant",
    "psnr_normality",
    "mse_normality",
    "ssim_ci",
    "ssim_significant",
    "ssim_normality",
    "png_path",
    "registry_key",
]


def create_metric_accumulators():
    """Create result metric accumulator lists keyed by logical metric name."""
    return {target: [] for _, target in METRIC_ACCUMULATOR_KEYS}


def append_result_metrics(result, accumulators):
    """Append all present result metrics to their aggregate lists."""
    for result_key, accumulator_key in METRIC_ACCUMULATOR_KEYS:
        value = result.get(result_key)
        if value is not None:
            accumulators[accumulator_key].append(value)


def print_result_summary(result):
    """Print per-experiment status lines matching the original script output."""
    es_r = result.get("early_stop_reason", {})
    es_i = result.get("early_stop_iter", {})
    print(f"early_stop iDLG: {es_r.get('iDLG')} @ {es_i.get('iDLG')}")
    if result.get("last_loss_iDLG") is not None:
        print("last_loss_iDLG:", result["last_loss_iDLG"], "last_mse_iDLG:", result["last_mse_iDLG"])
    if result.get("best_loss_iDLG") is not None:
        print("best_loss_iDLG:", result["best_loss_iDLG"], "best_mse_iDLG:", result["best_mse_iDLG"], "exp_idx:", result["idx_net"])
    if result.get("last_loss_iDLG_masked") is not None:
        print("last_loss_iDLG_masked:", result["last_loss_iDLG_masked"], "last_mse_iDLG_masked:", result["last_mse_iDLG_masked"])
    if result.get("best_loss_iDLG_masked") is not None:
        print("best_loss_iDLG_masked:", result["best_loss_iDLG_masked"], "best_mse_iDLG_masked:", result["best_mse_iDLG_masked"], "exp_idx:", result["idx_net"])
    if result.get("jac_rank_iDLG") is not None:
        print("jac_rank_iDLG:", result["jac_rank_iDLG"], "jac_shape_iDLG:", result["jac_shape_iDLG"])
    if result.get("jac_rank_iDLG_masked") is not None:
        print("jac_rank_iDLG_masked:", result["jac_rank_iDLG_masked"], "jac_shape_iDLG_masked:", result["jac_shape_iDLG_masked"])
    print("gt_label:", result["gt_label"],
          "lab_iDLG:", result["label_iDLG"], "lab_iDLG_masked:", result["label_iDLG_masked"])
    print("----------------------\n\n")


def _best_restart(rdict, sorted_idxs, key, pick="min"):
    candidates = [(i, rdict[i].get(key)) for i in sorted_idxs if rdict[i].get(key) is not None]
    if not candidates:
        return None
    fn = min if pick == "min" else max
    best_i, _ = fn(candidates, key=lambda x: x[1])
    return best_i


def _first(value):
    return value[0] if isinstance(value, (list, tuple)) and value else value


def running_best(restarts, value_key, mode, payload_key=None):
    """Return the running best value or payload over restart result dictionaries."""
    best_value = None
    best_payload = None
    out = []
    for i in sorted(restarts.keys()):
        value = _first(restarts[i].get(value_key))
        payload = _first(restarts[i].get(payload_key)) if payload_key else value
        better = (
            value is not None and payload is not None and
            (best_value is None or
             (mode == "min" and value < best_value) or
             (mode == "max" and value > best_value))
        )
        if better:
            best_value = value
            best_payload = payload
        out.append(best_payload)
    return out


def merge_restart_results(rdict):
    """Combine per-restart worker results for a single experiment."""
    sorted_idxs = sorted(rdict.keys())
    base = rdict[sorted_idxs[0]]

    best_idlg_i = _best_restart(rdict, sorted_idxs, "best_mse_iDLG", "min")
    best_masked_i = _best_restart(rdict, sorted_idxs, "best_mse_iDLG_masked", "min")
    best_idlg = rdict[best_idlg_i] if best_idlg_i is not None else base
    best_masked = rdict[best_masked_i] if best_masked_i is not None else base
    last_r = rdict[sorted_idxs[-1]]

    merged_recon = {}
    merged_recon.update(best_idlg.get("final_recon", {}))
    merged_recon.update(best_masked.get("final_recon", {}))

    return {
        "idx_net": base["idx_net"],
        "device_id": base["device_id"],
        "gt_data": base["gt_data"],
        "gt_label": base["gt_label"],
        "imidx_list": base["imidx_list"],
        "label_iDLG": base.get("label_iDLG"),
        "label_iDLG_masked": base.get("label_iDLG_masked"),
        "final_recon": merged_recon,
        "best_psnr_idlg": best_idlg.get("best_psnr_idlg"),
        "best_psnr_masked": best_masked.get("best_psnr_masked"),
        "best_loss_iDLG": best_idlg.get("best_loss_iDLG"),
        "best_mse_iDLG": best_idlg.get("best_mse_iDLG"),
        "best_loss_iDLG_masked": best_masked.get("best_loss_iDLG_masked"),
        "best_mse_iDLG_masked": best_masked.get("best_mse_iDLG_masked"),
        "best_ssim_idlg": best_idlg.get("best_ssim_idlg"),
        "best_ssim_masked": best_masked.get("best_ssim_masked"),
        "last_psnr_idlg": last_r.get("last_psnr_idlg"),
        "last_psnr_masked": last_r.get("last_psnr_masked"),
        "last_loss_iDLG": last_r.get("last_loss_iDLG"),
        "last_mse_iDLG": last_r.get("last_mse_iDLG"),
        "last_loss_iDLG_masked": last_r.get("last_loss_iDLG_masked"),
        "last_mse_iDLG_masked": last_r.get("last_mse_iDLG_masked"),
        "psnr_per_restart_idlg": running_best(rdict, "psnr_per_restart_idlg", "max"),
        "psnr_per_restart_masked": running_best(rdict, "psnr_per_restart_masked", "max"),
        "img_per_restart_idlg": running_best(rdict, "best_mse_iDLG", "min", "img_per_restart_idlg"),
        "img_per_restart_masked": running_best(rdict, "best_mse_iDLG_masked", "min", "img_per_restart_masked"),
        "mse_per_restart_idlg": running_best(rdict, "mse_per_restart_idlg", "min"),
        "mse_per_restart_masked": running_best(rdict, "mse_per_restart_masked", "min"),
        "ssim_per_restart_idlg": running_best(rdict, "mse_per_restart_idlg", "min", "ssim_per_restart_idlg"),
        "ssim_per_restart_masked": running_best(rdict, "mse_per_restart_masked", "min", "ssim_per_restart_masked"),
        "jac_rank_iDLG": best_idlg.get("jac_rank_iDLG"),
        "jac_shape_iDLG": best_idlg.get("jac_shape_iDLG"),
        "jac_rank_iDLG_masked": best_masked.get("jac_rank_iDLG_masked"),
        "jac_shape_iDLG_masked": best_masked.get("jac_shape_iDLG_masked"),
        "early_stop_reason": best_idlg.get("early_stop_reason", {}),
        "early_stop_iter": best_idlg.get("early_stop_iter", {}),
        "init_frames": best_idlg.get("init_frames", {}),
        "recon_frames": best_idlg.get("recon_frames", {}),
        "restart_idx": None,
    }


def empty_paired_report():
    """Return empty paired-comparison report values."""
    return {
        "mse_paired_stats": EMPTY_PAIRED_STATS.copy(),
        "psnr_paired_stats": EMPTY_PAIRED_STATS.copy(),
        "ssim_paired_stats": EMPTY_PAIRED_STATS.copy(),
        "mse_ci_str": "",
        "psnr_ci_str": "",
        "ssim_ci_str": "",
        "mse_significant_str": "",
        "psnr_significant_str": "",
        "ssim_significant_str": "",
        "psnr_normality_str": "",
        "mse_normality_str": "",
        "ssim_normality_str": "",
    }


def _report_from_paired_values(paired_values):
    summaries = paired_metric_summaries(paired_values)
    return {
        "mse_paired_stats": summaries["mse"]["stats"],
        "psnr_paired_stats": summaries["psnr"]["stats"],
        "ssim_paired_stats": summaries["ssim"]["stats"],
        "mse_ci_str": summaries["mse"]["ci_str"],
        "psnr_ci_str": summaries["psnr"]["ci_str"],
        "ssim_ci_str": summaries["ssim"]["ci_str"],
        "mse_significant_str": summaries["mse"]["significant_str"],
        "psnr_significant_str": summaries["psnr"]["significant_str"],
        "ssim_significant_str": summaries["ssim"]["significant_str"],
        "psnr_normality_str": summaries["psnr"]["normality_str"],
        "mse_normality_str": summaries["mse"]["normality_str"],
        "ssim_normality_str": summaries["ssim"]["normality_str"],
    }


def paired_report_for_both(all_results_by_idx, warn_fn):
    """Build paired stats from a run that contains both iDLG and masked results."""
    paired_values = {
        "mse_idlg": [],
        "mse_masked": [],
        "psnr_idlg": [],
        "psnr_masked": [],
        "ssim_idlg": [],
        "ssim_masked": [],
    }
    n_total = len(all_results_by_idx)
    for idx in sorted(all_results_by_idx):
        result = all_results_by_idx[idx]
        mse_idlg = result.get("best_mse_iDLG")
        mse_masked = result.get("best_mse_iDLG_masked")
        psnr_idlg = result.get("best_psnr_idlg")
        psnr_masked = result.get("best_psnr_masked")
        ssim_idlg = result.get("best_ssim_idlg")
        ssim_masked = result.get("best_ssim_masked")

        all_valid = (
            mse_idlg is not None and mse_masked is not None and
            psnr_idlg is not None and psnr_masked is not None and
            np.isfinite(mse_idlg) and np.isfinite(mse_masked) and
            np.isfinite(psnr_idlg) and np.isfinite(psnr_masked)
        )
        if not all_valid:
            warn_fn(f"[WARNING] Experiment {idx}: excluded from paired tests (non-finite MSE or PSNR).")
            continue
        paired_values["mse_idlg"].append(mse_idlg)
        paired_values["mse_masked"].append(mse_masked)
        paired_values["psnr_idlg"].append(psnr_idlg)
        paired_values["psnr_masked"].append(psnr_masked)
        paired_values["ssim_idlg"].append(ssim_idlg)
        paired_values["ssim_masked"].append(ssim_masked)

    if len(paired_values["psnr_masked"]) < n_total:
        print(f"WARNING: {n_total - len(paired_values['psnr_masked'])}/{n_total} "
              f"experiment(s) excluded from paired tests (non-finite values).")
    return _report_from_paired_values(paired_values)


def paired_report_for_masked(all_results_by_idx, baseline_entry, run_id):
    """Build paired stats for masked-only results against a stored iDLG baseline."""
    baseline_psnr_list = baseline_entry["best_psnr_list"]
    baseline_mse_list = baseline_entry["best_mse_list"]
    baseline_ssim_list = baseline_entry.get("best_ssim_list")
    baseline_start = int(baseline_entry.get("args", {}).get("run_id", 0))
    if baseline_ssim_list is None:
        print("WARNING: iDLG baseline does not have an SSIM list ('best_ssim_list'); skipping SSIM comparison.")
    n_total = len(all_results_by_idx)

    paired_values = {
        "mse_idlg": [],
        "mse_masked": [],
        "psnr_idlg": [],
        "psnr_masked": [],
        "ssim_idlg": [],
        "ssim_masked": [],
    }
    for idx in sorted(all_results_by_idx):
        result = all_results_by_idx[idx]
        baseline_idx = run_id - baseline_start + idx
        if baseline_idx < 0 or baseline_idx >= len(baseline_psnr_list):
            raise ValueError(
                f"No stored iDLG baseline for run_id={run_id + idx}. "
                f"Available baseline samples cover run_id={baseline_start}.."
                f"{baseline_start + len(baseline_psnr_list) - 1}."
            )

        psnr_baseline = baseline_psnr_list[baseline_idx]
        psnr_masked = result.get("best_psnr_masked")
        mse_baseline = baseline_mse_list[baseline_idx]
        mse_masked = result.get("best_mse_iDLG_masked")
        ssim_baseline = baseline_entry.get("best_ssim_by_run_id", {}).get(str(run_id + idx))
        if ssim_baseline is None and baseline_ssim_list is not None and baseline_idx < len(baseline_ssim_list):
            ssim_baseline = baseline_ssim_list[baseline_idx]
        ssim_masked = result.get("best_ssim_masked")

        all_valid = (
            psnr_masked is not None and mse_masked is not None and
            np.isfinite(psnr_masked) and np.isfinite(psnr_baseline) and
            np.isfinite(mse_masked) and np.isfinite(mse_baseline)
        )
        if not all_valid:
            print(f"WARNING: Experiment {idx} excluded from paired tests (non-finite values).")
            continue
        paired_values["psnr_idlg"].append(psnr_baseline)
        paired_values["psnr_masked"].append(psnr_masked)
        paired_values["mse_idlg"].append(mse_baseline)
        paired_values["mse_masked"].append(mse_masked)
        if (
            ssim_baseline is not None and ssim_masked is not None and
            np.isfinite(ssim_baseline) and np.isfinite(ssim_masked)
        ):
            paired_values["ssim_idlg"].append(ssim_baseline)
            paired_values["ssim_masked"].append(ssim_masked)

    n_included = len(paired_values["psnr_masked"])
    if n_included < n_total:
        print(f"WARNING: {n_total - n_included}/{n_total} experiment(s) excluded from paired tests (non-finite values).")

    report = _report_from_paired_values(paired_values)
    print(f"\nLoaded iDLG baseline (run_id={run_id}). Paired test uses {n_included}/{n_total} experiment(s).")
    return report


def compute_aggregate_stats(accumulators):
    """Compute all aggregate statistics printed and written by the batch runner."""
    return {
        "avg_psnr_idlg": mean_or_nan(accumulators["psnr_idlg"]),
        "avg_psnr_masked": mean_or_nan(accumulators["psnr_masked"]),
        "std_psnr_idlg": std_or_nan(accumulators["psnr_idlg"]),
        "std_psnr_masked": std_or_nan(accumulators["psnr_masked"]),
        "avg_final_loss_idlg": mean_or_nan(accumulators["final_loss_idlg"]),
        "avg_final_mse_idlg": mean_or_nan(accumulators["final_mse_idlg"]),
        "avg_final_loss_masked": mean_or_nan(accumulators["final_loss_masked"]),
        "avg_final_mse_masked": mean_or_nan(accumulators["final_mse_masked"]),
        "med_final_loss_idlg": median_or_nan(accumulators["final_loss_idlg"]),
        "med_final_mse_idlg": median_or_nan(accumulators["final_mse_idlg"]),
        "med_final_loss_masked": median_or_nan(accumulators["final_loss_masked"]),
        "med_final_mse_masked": median_or_nan(accumulators["final_mse_masked"]),
        "avg_best_psnr_idlg": mean_or_nan(accumulators["best_psnr_idlg"]),
        "avg_best_psnr_masked": mean_or_nan(accumulators["best_psnr_masked"]),
        "std_best_psnr_idlg": std_or_nan(accumulators["best_psnr_idlg"]),
        "std_best_psnr_masked": std_or_nan(accumulators["best_psnr_masked"]),
        "avg_best_loss_idlg": mean_or_nan(accumulators["best_loss_idlg"]),
        "avg_best_mse_idlg": mean_or_nan(accumulators["best_mse_idlg"]),
        "avg_best_loss_masked": mean_or_nan(accumulators["best_loss_masked"]),
        "avg_best_mse_masked": mean_or_nan(accumulators["best_mse_masked"]),
        "med_best_loss_idlg": median_or_nan(accumulators["best_loss_idlg"]),
        "med_best_mse_idlg": median_or_nan(accumulators["best_mse_idlg"]),
        "med_best_loss_masked": median_or_nan(accumulators["best_loss_masked"]),
        "med_best_mse_masked": median_or_nan(accumulators["best_mse_masked"]),
        "avg_best_ssim_idlg": mean_or_nan(accumulators["best_ssim_idlg"]),
        "avg_best_ssim_masked": mean_or_nan(accumulators["best_ssim_masked"]),
        "std_best_ssim_idlg": std_or_nan(accumulators["best_ssim_idlg"]),
        "std_best_ssim_masked": std_or_nan(accumulators["best_ssim_masked"]),
    }


def ordered_idlg_baseline_lists(all_results_by_idx):
    """Return ordered baseline PSNR, MSE, and SSIM lists for registry storage."""
    ordered_best_psnr_idlg = []
    ordered_best_mse_idlg = []
    ordered_best_ssim_idlg = []

    for idx in sorted(all_results_by_idx):
        result = all_results_by_idx[idx]
        psnr = result.get("best_psnr_idlg")
        mse = result.get("best_mse_iDLG")
        ssim = result.get("best_ssim_idlg")

        if psnr is None or mse is None or not np.isfinite(psnr) or not np.isfinite(mse):
            raise ValueError(f"Missing or invalid iDLG baseline result for experiment idx={idx}")

        ordered_best_psnr_idlg.append(float(psnr))
        ordered_best_mse_idlg.append(float(mse))
        ordered_best_ssim_idlg.append(float(ssim) if ssim is not None and np.isfinite(ssim) else float("nan"))

    return ordered_best_psnr_idlg, ordered_best_mse_idlg, ordered_best_ssim_idlg


def ordered_masked_registry_lists(all_results_by_idx):
    """Return masked PSNR, MSE, and SSIM lists ordered by experiment index."""
    ordered_best_psnr_masked = []
    ordered_best_mse_masked = []
    ordered_best_ssim_masked = []

    for idx in sorted(all_results_by_idx):
        result = all_results_by_idx[idx]
        psnr = result.get("best_psnr_masked")
        mse = result.get("best_mse_iDLG_masked")
        ssim = result.get("best_ssim_masked")

        if psnr is None or mse is None or not np.isfinite(psnr) or not np.isfinite(mse):
            raise ValueError(f"Missing or invalid masked result for experiment idx={idx}")

        ordered_best_psnr_masked.append(float(psnr))
        ordered_best_mse_masked.append(float(mse))
        ordered_best_ssim_masked.append(float(ssim) if ssim is not None and np.isfinite(ssim) else float("nan"))

    return ordered_best_psnr_masked, ordered_best_mse_masked, ordered_best_ssim_masked


def grad_param_value(mask_mode, gradsize_topfrac, gradsize_topk):
    """Return the CSV grad_param value for the active mask mode."""
    if mask_mode in [
        "gradsize_topfrac",
        "gradsize_topfrac_entries",
        "gradsize_topfrac_entries_layer",
        "prefix_topfrac",
        "prefix_topfrac_entries",
        "prefix_topfrac_entries_layer",
    ]:
        return gradsize_topfrac
    if mask_mode in [
        "gradsize_topk",
        "gradsize_topk_entries",
        "gradsize_topk_entries_layer",
        "prefix_topk",
        "prefix_topk_entries",
        "prefix_topk_entries_layer",
    ]:
        return gradsize_topk
    return ""


def build_common_csv_fields(timestamp_str, argv, dataset, network_name, network_trained,
                            num_restarts, lr, gamma, iteration, num_exp,
                            tv_weight, optimizer, max_iteration, history_size):
    """Build common experiment CSV fields in the original field order."""
    return {
        "timestamp": timestamp_str,
        "job_id": "INTERACTIVE" if os.environ.get("LSB_INTERACTIVE") == "Y" else os.environ.get("LSB_JOBID", ""),
        "Run by": os.environ.get("USER", ""),
        "device": os.environ.get("LSB_QUEUE", ""),
        "cmd": "python " + " ".join(argv),
        "dataset": dataset,
        "network": network_name,
        "pretrained": network_trained,
        "restarts": num_restarts,
        "lr": lr,
        "gamma": gamma,
        "iteration": iteration,
        "num_exp": num_exp,
        "tv_weight": tv_weight,
        "optimizer": optimizer,
        "max_iter": max_iteration,
        "history": history_size,
    }


def build_exp_result_rows(methods, common, stats, paired_report, baseline_key,
                          masked_key, mask_mode, prefixes, grad_value, png_path_str):
    """Build experiment result rows while preserving existing column names and values."""
    rows = []

    if methods in ["idlg", "both"]:
        rows.append({
            "method": "iDLG",
            **common,
            "registry_key": baseline_key,
            "mask_mode": "",
            "grad_param": "",
            "med_best_loss": round(stats["med_best_loss_idlg"], 5),
            "avg_best_loss": round(stats["avg_best_loss_idlg"], 5),
            "med_best_mse": round(stats["med_best_mse_idlg"], 10),
            "avg_best_mse": round(stats["avg_best_mse_idlg"], 10),
            "avg_best_psnr": round(stats["avg_best_psnr_idlg"], 5),
            "std_best_psnr": round(stats["std_best_psnr_idlg"], 5),
            "avg_best_ssim": round(stats["avg_best_ssim_idlg"], 5),
            "std_best_ssim": round(stats["std_best_ssim_idlg"], 5),
            "png_path": png_path_str,
        })

    if methods in ["masked", "both"]:
        rows.append({
            "method": "iDLG_masked",
            **common,
            "registry_key": masked_key or "",
            "mask_mode": mask_mode,
            "prefixes": prefixes if "prefix" in mask_mode else "",
            "grad_param": grad_value,
            "mse_ci": paired_report["mse_ci_str"] if methods in ["both", "masked"] else "",
            "mse_significant": paired_report["mse_significant_str"] if methods in ["both", "masked"] else "",
            "psnr_ci": paired_report["psnr_ci_str"] if methods in ["both", "masked"] else "",
            "psnr_significant": paired_report["psnr_significant_str"] if methods in ["both", "masked"] else "",
            "psnr_normality": paired_report["psnr_normality_str"] if methods in ["both", "masked"] else "",
            "mse_normality": paired_report["mse_normality_str"] if methods in ["both", "masked"] else "",
            "ssim_ci": paired_report["ssim_ci_str"] if methods in ["both", "masked"] else "",
            "ssim_significant": paired_report["ssim_significant_str"] if methods in ["both", "masked"] else "",
            "ssim_normality": paired_report["ssim_normality_str"] if methods in ["both", "masked"] else "",
            "med_best_loss": round(stats["med_best_loss_masked"], 5),
            "avg_best_loss": round(stats["avg_best_loss_masked"], 5),
            "med_best_mse": round(stats["med_best_mse_masked"], 10),
            "avg_best_mse": round(stats["avg_best_mse_masked"], 10),
            "avg_best_psnr": round(stats["avg_best_psnr_masked"], 5),
            "std_best_psnr": round(stats["std_best_psnr_masked"], 5),
            "avg_best_ssim": round(stats["avg_best_ssim_masked"], 5),
            "std_best_ssim": round(stats["std_best_ssim_masked"], 5),
            "png_path": png_path_str,
        })

    fieldnames = ["method"] + list(common.keys()) + EXP_RESULT_EXTRA_FIELDS
    return rows, fieldnames


def write_masking_sweep_mse_csv(save_path, network_name, dataset, mask_mode, masked_key,
                                masked_comparable_args, gradsize_topfrac, argv):
    """Append the masking-sweep MSE row for per-layer top-fraction runs."""
    if masked_key is None:
        print("\n[WARNING] No masked registry key available for masking-sweep CSV.")
        return

    sweep_file_args = dict(masked_comparable_args)
    for drop_key in ("gradsize_topfrac", "gradsize_topk", "run_id", "num_exp"):
        sweep_file_args.pop(drop_key, None)
    key_json = json.dumps(sweep_file_args, sort_keys=True, separators=(",", ":"))
    key_hash = hashlib.md5(key_json.encode("utf-8")).hexdigest()[:12]
    sweep_dir = os.path.join(save_path, "masking_sweeps")
    csv_path_sweep = os.path.join(
        sweep_dir,
        f"mse_{network_name}_{dataset}_{mask_mode}_{key_hash}.csv",
    )
    row = {
        "command": "python " + " ".join(argv),
        "topfrac": gradsize_topfrac,
        "masked_key": masked_key,
    }
    fields = [
        "command",
        "topfrac",
        "masked_key",
    ]

    append_csv_row(csv_path_sweep, row, fields)
    print(f"\nSaved masking-sweep MSE row to: {csv_path_sweep}")


def print_final_experiment_summary(csv_path, gamma, mask_mode, gradsize_topfrac,
                                   gradsize_topk, stats, paired_report, methods):
    """Print the final aggregate experiment summary."""
    print("\n=== Average PSNR over all experiments ===")
    print(f"\nSaved CSV rows to: {csv_path}")
    print(f"Gamma for lr scheduler was: {gamma}")
    if mask_mode in ("gradsize_topfrac", "gradsize_topfrac_entries", "gradsize_topfrac_entries_layer"):
        print(f"top fraction kept ({mask_mode}): {gradsize_topfrac*100}%")
    elif mask_mode in ("gradsize_topk", "gradsize_topk_entries", "gradsize_topk_entries_layer"):
        print(f"top-k kept ({mask_mode}): {gradsize_topk}")
    print(f"Avg final loss iDLG: {stats['avg_final_loss_idlg']:.6f} | masked: {stats['avg_final_loss_masked']:.6f}")
    print(f"Avg final mse  iDLG: {stats['avg_final_mse_idlg']:.8f} | masked: {stats['avg_final_mse_masked']:.8f}")
    print(f"Median final loss iDLG: {stats['med_final_loss_idlg']:.6f} | masked: {stats['med_final_loss_masked']:.6f}")
    print(f"Median final mse  iDLG: {stats['med_final_mse_idlg']:.8f} | masked: {stats['med_final_mse_masked']:.8f}")
    print(f"Average PSNR iDLG: {stats['avg_psnr_idlg']:.4f} ± {stats['std_psnr_idlg']:.4f} dB | masked: {stats['avg_psnr_masked']:.4f} ± {stats['std_psnr_masked']:.4f} dB")
    print(f"Avg best loss iDLG: {stats['avg_best_loss_idlg']:.6f} | masked: {stats['avg_best_loss_masked']:.6f}")
    print(f"Avg best mse  iDLG: {stats['avg_best_mse_idlg']:.8f} | masked: {stats['avg_best_mse_masked']:.8f}")
    print(f"Median best loss iDLG: {stats['med_best_loss_idlg']:.6f} | masked: {stats['med_best_loss_masked']:.6f}")
    print(f"Median best mse  iDLG: {stats['med_best_mse_idlg']:.10f} | masked: {stats['med_best_mse_masked']:.10f}")
    print(f"Average best PSNR iDLG: {stats['avg_best_psnr_idlg']:.4f} ± {stats['std_best_psnr_idlg']:.4f} dB | masked: {stats['avg_best_psnr_masked']:.4f} ± {stats['std_best_psnr_masked']:.4f} dB")
    print(f"Average best SSIM iDLG: {stats['avg_best_ssim_idlg']:.4f} ± {stats['std_best_ssim_idlg']:.4f} | masked: {stats['avg_best_ssim_masked']:.4f} ± {stats['std_best_ssim_masked']:.4f}")
    if methods in ["both", "masked"]:
        print(
            f"Paired best MSE diff (masked - iDLG): {paired_report['mse_paired_stats']['mean_diff']:.10f} "
            f"| 95% CI: {paired_report['mse_ci_str']} "
            f"| significant: {paired_report['mse_significant_str']} "
            f"| normality: {paired_report['mse_normality_str']}")
        print(
            f"Paired best PSNR diff (masked - iDLG): {paired_report['psnr_paired_stats']['mean_diff']:.5f} dB "
            f"| 95% CI: {paired_report['psnr_ci_str']} "
            f"| significant: {paired_report['psnr_significant_str']} "
            f"| normality: {paired_report['psnr_normality_str']}")
        print(
            f"Paired best SSIM diff (masked - iDLG): {paired_report['ssim_paired_stats']['mean_diff']:.5f} "
            f"| 95% CI: {paired_report['ssim_ci_str']} "
            f"| significant: {paired_report['ssim_significant_str']} "
            f"| normality: {paired_report['ssim_normality_str']}")


class ResultAggregator:
    """Accumulates per-experiment result metrics and computes aggregate statistics.

    Thin façade over the module-level accumulator helpers: holds the dict produced
    by create_metric_accumulators() and delegates to append_result_metrics() /
    compute_aggregate_stats().
    """

    def __init__(self):
        """Initialise empty metric accumulator lists for all tracked metrics."""
        self.accumulators = create_metric_accumulators()

    def append(self, result):
        """Append one worker result's metrics to the accumulators."""
        append_result_metrics(result, self.accumulators)

    def aggregate_stats(self):
        """Return the aggregate-statistics dict for the accumulated results."""
        return compute_aggregate_stats(self.accumulators)


class RestartSelector:
    """Selects and merges the best per-restart worker results for one experiment."""

    @staticmethod
    def merge(rdict):
        """Combine per-restart worker results into a single experiment result dict."""
        return merge_restart_results(rdict)
