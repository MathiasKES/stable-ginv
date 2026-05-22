import csv
import hashlib
import json
import math
import os

import numpy as np
from scipy import stats


def paired_t_ci(x, y, confidence=0.95):
    """Paired t-test CI for mean(x - y); returns dict with mean_diff, std_diff, ci, t_stat, p_value."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if len(x) != len(y):
        raise ValueError("Paired CI requires arrays of equal length.")
    if len(x) < 2:
        return {
            "n": len(x),
            "mean_diff": float("nan"),
            "std_diff": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "t_stat": float("nan"),
            "p_value": float("nan"),
        }

    d = x - y
    n = len(d)
    mean_diff = float(np.mean(d))
    std_diff = float(np.std(d, ddof=1))
    se = std_diff / math.sqrt(n)

    alpha = 1.0 - confidence
    tcrit = stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)

    ci_low = mean_diff - tcrit * se
    ci_high = mean_diff + tcrit * se

    t_stat, p_value = stats.ttest_rel(x, y)

    return {
        "n": n,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
    }


def paired_summary(x_masked, x_idlg, metric, confidence=0.95, ci_decimals=5):
    """Summarise paired comparison; returns dict with stats, ci_str, and significant_str."""
    result = paired_t_ci(x_masked, x_idlg, confidence=confidence)

    if np.isnan(result["ci_low"]) or np.isnan(result["ci_high"]):
        return {
            "stats": result,
            "ci_str": "",
            "significant_str": "",
        }

    significant = not (result["ci_low"] <= 0 <= result["ci_high"])

    if not significant:
        better = ""
    elif metric == "mse":
        better = "masked" if result["mean_diff"] < 0 else "idlg"
    elif metric == "psnr":
        better = "masked" if result["mean_diff"] > 0 else "idlg"
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return {
        "stats": result,
        "ci_str": f"[{result['ci_low']:.{ci_decimals}f}, {result['ci_high']:.{ci_decimals}f}]",
        "significant_str": f"True, {better}" if significant else "False",
    }


def baseline_key_from_args(args):
    """Hash of reconstruction/data/model hyperparameters for paired baseline comparison."""
    comparable = {
        "dataset": args.dataset,
        "network": args.network,
        "pretrained": bool(args.pretrained),
        "lr": args.lr,
        "gamma": args.gamma,
        "grad_loss": args.grad_loss,
        "num_dummy": args.num_dummy,
        "iteration": args.iteration,
        "num_exp": args.num_exp,
        "run_id": args.run_id,
        "tv_weight": args.tv_weight,
        "optimizer": args.optimizer,
        "num_restarts": args.num_restarts,
        "max_iteration": args.max_iteration,
        "history_size": args.history_size,
    }

    key_json = json.dumps(comparable, sort_keys=True)
    key_hash = hashlib.md5(key_json.encode("utf-8")).hexdigest()
    return key_hash, comparable


def load_baseline_registry(path):
    """Load JSON baseline registry from path; returns empty dict if file does not exist."""
    if not os.path.isfile(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_baseline_registry(path, registry):
    """Write baseline registry dict to JSON at path."""
    os.makedirs(os.path.dirname(path), mode=0o770, exist_ok=True)
    with open(path, "w") as f:
        json.dump(registry, f, indent=2)


def update_idlg_baseline(registry, key, comparable_args, best_psnr_list, best_mse_list):
    """Insert or incrementally average a new run into the baseline registry."""
    if key not in registry:
        registry[key] = {
            "args": comparable_args,
            "num_runs_averaged": 1,
            "avg_best_psnr_list": best_psnr_list,
            "avg_best_mse_list": best_mse_list,
        }
        return registry[key]

    entry = registry[key]

    old_n = entry["num_runs_averaged"]
    new_n = old_n + 1

    old_psnr = np.array(entry["avg_best_psnr_list"], dtype=float)
    old_mse = np.array(entry["avg_best_mse_list"], dtype=float)

    new_psnr = np.array(best_psnr_list, dtype=float)
    new_mse = np.array(best_mse_list, dtype=float)

    if len(old_psnr) != len(new_psnr):
        raise ValueError(
            f"Baseline with same arguments has different num_exp length: "
            f"old={len(old_psnr)}, new={len(new_psnr)}"
        )

    entry["avg_best_psnr_list"] = ((old_psnr * old_n + new_psnr) / new_n).tolist()
    entry["avg_best_mse_list"] = ((old_mse * old_n + new_mse) / new_n).tolist()
    entry["num_runs_averaged"] = new_n

    return entry


def write_baseline_summary_csv(path, registry):
    """Write a summary CSV of all baseline registry entries."""
    os.makedirs(os.path.dirname(path), mode=0o770, exist_ok=True)

    fieldnames = [
        "baseline_key",
        "num_runs_averaged",
        "dataset",
        "network",
        "pretrained",
        "lr",
        "gamma",
        "grad_loss",
        "num_dummy",
        "iteration",
        "num_exp",
        "run_id",
        "tv_weight",
        "optimizer",
        "num_restarts",
        "max_iteration",
        "history_size",
        "avg_best_psnr",
        "std_best_psnr",
        "avg_best_mse",
        "avg_best_psnr_list",
        "avg_best_mse_list",
    ]

    rows = []
    for key, entry in registry.items():
        a = entry["args"]
        psnr = np.array(entry["avg_best_psnr_list"], dtype=float)
        mse = np.array(entry["avg_best_mse_list"], dtype=float)

        rows.append({
            "baseline_key": key,
            "num_runs_averaged": entry["num_runs_averaged"],
            **a,
            "avg_best_psnr": float(np.mean(psnr)) if len(psnr) else float("nan"),
            "std_best_psnr": float(np.std(psnr, ddof=1)) if len(psnr) > 1 else float("nan"),
            "avg_best_mse": float(np.mean(mse)) if len(mse) else float("nan"),
            "avg_best_psnr_list": json.dumps(entry["avg_best_psnr_list"]),
            "avg_best_mse_list": json.dumps(entry["avg_best_mse_list"]),
        })

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
