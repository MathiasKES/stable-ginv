import csv
import hashlib
import json
import math
import os

import numpy as np
from scipy import stats


def parse_prefixes_with_fracs(prefixes_str):
    """Parse 'conv1:0.5,layer1:1.0,fc' into (prefixes_tuple, fracs_dict)."""
    prefixes = []
    fracs = {}
    for item in prefixes_str.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            prefix, frac = item.split(":", 1)
            prefix = prefix.strip()
            fracs[prefix] = float(frac.strip())
            prefixes.append(prefix)
        else:
            prefixes.append(item)
    return tuple(prefixes), fracs


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
    """Store a single iDLG baseline run. Overwrites any existing entry for the same key."""
    if key in registry:
        print(
            f"\nWARNING: Overwriting existing iDLG baseline for "
            f"run_id={comparable_args['run_id']}."
        )

    entry = {
        "args": comparable_args,
        "best_psnr_list": best_psnr_list,
        "best_mse_list": best_mse_list,
    }
    registry[key] = entry
    return entry


def write_baseline_summary_csv(path, registry):
    """Write a summary CSV of all baseline registry entries."""
    os.makedirs(os.path.dirname(path), mode=0o770, exist_ok=True)

    fieldnames = [
        "baseline_key",
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
        "best_psnr_list",
        "best_mse_list",
    ]

    rows = []
    for key, entry in registry.items():
        a = entry["args"]
        psnr = np.array(entry["best_psnr_list"], dtype=float)
        mse = np.array(entry["best_mse_list"], dtype=float)

        rows.append({
            "baseline_key": key,
            **a,
            "avg_best_psnr": float(np.mean(psnr)) if len(psnr) else float("nan"),
            "std_best_psnr": float(np.std(psnr, ddof=1)) if len(psnr) > 1 else float("nan"),
            "avg_best_mse": float(np.mean(mse)) if len(mse) else float("nan"),
            "best_psnr_list": json.dumps(entry["best_psnr_list"]),
            "best_mse_list": json.dumps(entry["best_mse_list"]),
        })

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
