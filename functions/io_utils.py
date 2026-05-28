import csv
import hashlib
import json
import math
import os

import sys
from datetime import datetime

import numpy as np
from scipy import stats


def safe_chmod(path, mode=0o770):
    """Set chmod on path, swallowing and logging errors."""
    try:
        os.chmod(path, mode)
    except OSError as e:
        print(f"[WARNING] Failed to chmod {path}: {e}")


def safe_makedirs(path, mode=0o770):
    """os.makedirs(exist_ok=True) with error swallowing. Returns True on success."""
    if not path:
        return True
    try:
        os.makedirs(path, mode=mode, exist_ok=True)
        return True
    except OSError as e:
        print(f"[WARNING] Failed to create directory {path}: {e}")
        return False


def safe_write(path, writer_fn, mode="w", newline=None, chmod_mode=0o770):
    """Open path for writing, call writer_fn(file). Best-effort chmod 770 on parent dir and file.
    Logs and returns False on any OSError instead of raising. Returns True on success.
    """
    parent = os.path.dirname(path)
    if parent and not safe_makedirs(parent, mode=chmod_mode):
        return False
    open_kwargs = {} if newline is None else {"newline": newline}
    try:
        with open(path, mode, **open_kwargs) as f:
            writer_fn(f)
    except OSError as e:
        print(f"[WARNING] Failed to write {path}: {e}")
        return False
    safe_chmod(path, mode=chmod_mode)
    return True


def safe_savefig(fig, path, chmod_mode=0o770, **savefig_kwargs):
    """Save matplotlib figure to path with best-effort chmod 770. Returns True on success."""
    parent = os.path.dirname(path)
    if parent and not safe_makedirs(parent, mode=chmod_mode):
        return False
    try:
        fig.savefig(path, **savefig_kwargs)
    except OSError as e:
        print(f"[WARNING] Failed to save figure {path}: {e}")
        return False
    safe_chmod(path, mode=chmod_mode)
    return True


def setstdout(ts=None, path=None):
    """Set up stdout tee to a log file. Returns the path used, or None if not interactive.

    ts:   timestamp string to name a new log file (ignored if path is given).
    path: full path to an existing log file to append to (worker processes pass this).
    If neither is given, a new file is created using datetime.now().
    """
    if os.environ.get("LSB_INTERACTIVE", default="N") != "Y":
        return None

    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for stream in self.streams:
                stream.write(data)
                stream.flush()

        def flush(self):
            for stream in self.streams:
                stream.flush()

    terminal = sys.stdout

    if path is None:
        if ts is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if os.path.exists("/work3/s234843/bachelor/gpuout/idlg"):
            path = f"/work3/s234843/bachelor/gpuout/idlg/i{ts}.out"
        else:
            safe_makedirs("./gpuout")
            path = f"./gpuout/i{ts}.out"

    try:
        logfile = open(path, "a")
    except OSError as e:
        print(f"[WARNING] Failed to open stdout log {path}: {e}")
        return None
    safe_chmod(path)
    sys.stdout = Tee(terminal, logfile)
    return path

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

    if n >= 3:
        sw_stat, sw_p = stats.shapiro(d)
    else:
        sw_stat, sw_p = float("nan"), float("nan")

    return {
        "n": n,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "shapiro_stat": float(sw_stat),
        "shapiro_p": float(sw_p),
    }


def paired_summary(x_masked, x_idlg, metric, confidence=0.95, ci_decimals=5):
    """Summarise paired comparison; returns dict with stats, ci_str, and significant_str."""
    result = paired_t_ci(x_masked, x_idlg, confidence=confidence)

    if np.isnan(result["ci_low"]) or np.isnan(result["ci_high"]):
        return {
            "stats": result,
            "ci_str": "",
            "significant_str": "",
            "normality_str": "n/a",
        }

    significant = not (result["ci_low"] <= 0 <= result["ci_high"])

    if not significant:
        better = ""
    elif metric == "mse":
        better = "masked" if result["mean_diff"] < 0 else "idlg"
    elif metric in ("psnr", "ssim"):
        better = "masked" if result["mean_diff"] > 0 else "idlg"
    else:
        raise ValueError(f"Unknown metric: {metric}")

    sw_p = result["shapiro_p"]
    if np.isnan(sw_p):
        normality_str = "n/a"
    elif sw_p > 0.05:
        normality_str = f"normal (W={result['shapiro_stat']:.4f}, p={sw_p:.4f})"
    else:
        normality_str = f"NON-NORMAL (W={result['shapiro_stat']:.4f}, p={sw_p:.4f})"

    return {
        "stats": result,
        "ci_str": f"[{result['ci_low']:.{ci_decimals}f}, {result['ci_high']:.{ci_decimals}f}]",
        "significant_str": f"True, {better}" if significant else "False",
        "normality_str": normality_str,
    }


def masked_key_from_args(args):
    """Hash of all hyperparameters (reconstruction + masking) for the masked registry."""
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
        "mask_mode": args.mask_mode,
        "gradsize_topk": args.gradsize_topk,
        "gradsize_topfrac": args.gradsize_topfrac,
        "gradsize_metric": args.gradsize_metric,
        "prefixes": args.prefixes,
    }
    key_json = json.dumps(comparable, sort_keys=True)
    key_hash = hashlib.md5(key_json.encode("utf-8")).hexdigest()
    return key_hash, comparable


def load_masked_registry(path):
    """Load JSON masked registry from path; returns empty dict if file does not exist."""
    if not os.path.isfile(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_masked_registry(path, registry):
    """Write masked registry dict to JSON at path. Failures are logged, not raised."""
    return safe_write(path, lambda f: json.dump(registry, f, indent=2))


def update_masked_registry(registry, key, comparable_args, best_psnr_list, best_mse_list, best_ssim_list):
    """Store a single masked run entry. Overwrites any existing entry for the same key."""
    if key in registry:
        print(
            f"\nWARNING: Overwriting existing masked registry entry for "
            f"run_id={comparable_args.get('run_id')}, mask_mode={comparable_args.get('mask_mode')}."
        )
    registry[key] = {
        "args": comparable_args,
        "best_psnr_list": [float(v) for v in best_psnr_list],
        "best_mse_list": [float(v) for v in best_mse_list],
        "best_ssim_list": [float(v) for v in best_ssim_list],
    }
    return registry[key]


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
    """Write baseline registry dict to JSON at path. Failures are logged, not raised."""
    return safe_write(path, lambda f: json.dump(registry, f, indent=2))


def update_idlg_baseline(registry, key, comparable_args, best_psnr_list, best_mse_list, best_ssim_list):
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
        "best_ssim_list": best_ssim_list,
    }
    registry[key] = entry
    return entry


def write_baseline_summary_csv(path, registry):
    """Write a summary CSV of all baseline registry entries. Failures are logged, not raised."""
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
        "avg_best_ssim",
        "std_best_ssim",
        "best_psnr_list",
        "best_mse_list",
        "best_ssim_list",
    ]

    rows = []
    for key, entry in registry.items():
        a = entry["args"]
        psnr = np.array(entry["best_psnr_list"], dtype=float)
        mse = np.array(entry["best_mse_list"], dtype=float)
        ssim_list = entry.get("best_ssim_list", [])
        ssim = np.array(ssim_list, dtype=float)

        rows.append({
            "baseline_key": key,
            **a,
            "avg_best_psnr": float(np.mean(psnr)) if len(psnr) else float("nan"),
            "std_best_psnr": float(np.std(psnr, ddof=1)) if len(psnr) > 1 else float("nan"),
            "avg_best_mse": float(np.mean(mse)) if len(mse) else float("nan"),
            "avg_best_ssim": float(np.mean(ssim)) if len(ssim) else float("nan"),
            "std_best_ssim": float(np.std(ssim, ddof=1)) if len(ssim) > 1 else float("nan"),
            "best_psnr_list": json.dumps(entry["best_psnr_list"]),
            "best_mse_list": json.dumps(entry["best_mse_list"]),
            "best_ssim_list": json.dumps(ssim_list),
        })

    def _write(f):
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return safe_write(path, _write, newline="")
