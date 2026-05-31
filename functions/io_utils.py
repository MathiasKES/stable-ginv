import csv
import copy
import hashlib
import json
import math
import os
import sys
from datetime import datetime
from statistics import NormalDist

import numpy as np

_SCIPY_STATS = None
_SCIPY_STATS_IMPORT_ERROR = None


def _load_scipy_stats():
    """Import scipy.stats only when needed; return None when the HPC runtime cannot load it."""
    global _SCIPY_STATS, _SCIPY_STATS_IMPORT_ERROR
    if _SCIPY_STATS is not None:
        return _SCIPY_STATS
    if _SCIPY_STATS_IMPORT_ERROR is not None:
        return None
    try:
        from scipy import stats as scipy_stats
    except Exception as exc:
        _SCIPY_STATS_IMPORT_ERROR = exc
        print(
            "[WARNING] scipy.stats could not be imported; paired p-values and "
            f"Shapiro normality tests will be skipped. Import error: {exc}"
        )
        return None
    _SCIPY_STATS = scipy_stats
    return _SCIPY_STATS


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
    except Exception as e:
        print(f"[WARNING] Failed to save figure {path}: {e}")
        return False
    safe_chmod(path, mode=chmod_mode)
    return True


def resolve_storage_paths(root_path="."):
    """Return (data_path, save_path) for DTU HPC when available, otherwise local paths."""
    if os.access("/work3/s234843/bachelor", os.R_OK | os.W_OK | os.X_OK):
        return "/work3/s234843/bachelor/datasets", "/work3/s234843/bachelor/results"
    data_path = os.path.join(root_path, "data").replace("\\", "/")
    save_path = os.path.join(root_path, "results").replace("\\", "/")
    return data_path, save_path


def mean_or_nan(values):
    """Return float mean, or nan for an empty sequence."""
    return float(np.mean(values)) if len(values) else float("nan")


def std_or_nan(values, ddof=1):
    """Return float std when there are enough samples for ddof, otherwise nan."""
    return float(np.std(values, ddof=ddof)) if len(values) > ddof else float("nan")


def median_or_nan(values):
    """Return float median, or nan for an empty sequence."""
    return float(np.median(values)) if len(values) else float("nan")


def append_csv_rows(path, rows, fieldnames):
    """Append rows to a CSV file, writing the header when the file is new."""
    file_exists = os.path.isfile(path)

    def _append(f):
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    return safe_write(path, _append, mode="a", newline="")


def append_csv_row(path, row, fieldnames):
    """Append one row to a CSV file, writing the header when the file is new."""
    return append_csv_rows(path, [row], fieldnames)


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

    scipy_stats = _load_scipy_stats()
    alpha = 1.0 - confidence
    if scipy_stats is None:
        tcrit = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    else:
        tcrit = scipy_stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)

    ci_low = mean_diff - tcrit * se
    ci_high = mean_diff + tcrit * se

    if scipy_stats is None:
        t_stat, p_value = float("nan"), float("nan")
    else:
        t_stat, p_value = scipy_stats.ttest_rel(x, y)

    if scipy_stats is not None and n >= 3:
        sw_stat, sw_p = scipy_stats.shapiro(d)
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


def normality_str_from_ci(ci):
    """Format Shapiro normality details from a CI result dict."""
    sw_p = ci.get("shapiro_p", float("nan"))
    if np.isnan(sw_p):
        return "normality: n/a"
    label = "normal" if sw_p > 0.05 else "NON-NORMAL"
    return f"normality: {label} (W={ci['shapiro_stat']:.4f}, p={sw_p:.4f})"


def paired_metric_summaries(paired_values):
    """Return paired-summary fields for MSE, PSNR, and SSIM metric lists."""
    empty_stats = {
        "n": float("nan"),
        "mean_diff": float("nan"),
        "std_diff": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "t_stat": float("nan"),
        "p_value": float("nan"),
    }
    out = {}
    for metric, decimals in (("mse", 10), ("psnr", 5), ("ssim", 5)):
        masked_key = f"{metric}_masked"
        idlg_key = f"{metric}_idlg"
        if paired_values.get(masked_key):
            summary = paired_summary(
                np.array(paired_values[masked_key]),
                np.array(paired_values[idlg_key]),
                metric=metric,
                confidence=0.95,
                ci_decimals=decimals,
            )
        else:
            summary = {
                "stats": empty_stats.copy(),
                "ci_str": "",
                "significant_str": "",
                "normality_str": "",
            }
        out[metric] = summary
    return out


def masked_key_from_args(args):
    """Hash of masking hyperparameters, excluding the sample range for appendable runs."""
    comparable = {
        "dataset": args.dataset,
        "network": args.network,
        "pretrained": bool(args.pretrained),
        "lr": args.lr,
        "gamma": args.gamma,
        "grad_loss": args.grad_loss,
        "num_dummy": args.num_dummy,
        "iteration": args.iteration,
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
    comparable["num_exp"] = args.num_exp
    comparable["run_id"] = args.run_id
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


def _config_without_sample_range(args):
    return {
        name: value
        for name, value in args.items()
        if name not in ("num_exp", "run_id")
    }


def find_registry_entry(registry, key, comparable_args):
    """Return a current or legacy registry entry matching one configuration."""
    if key in registry:
        return key, registry[key]
    target_config = _config_without_sample_range(comparable_args)
    matches = [
        (stored_key, entry)
        for stored_key, entry in registry.items()
        if _config_without_sample_range(entry.get("args", {})) == target_config
    ]
    if len(matches) > 1:
        raise ValueError(
            "Multiple legacy registry entries match this configuration. "
            "Merge or remove the duplicate entries before appending."
        )
    return matches[0] if matches else (None, None)


def seed_registry_entry_from_fallback(registry, fallback_registry, key, comparable_args):
    """Copy one matching read-only fallback entry into a writable registry."""
    if find_registry_entry(registry, key, comparable_args)[1] is not None:
        return
    _, entry = find_registry_entry(fallback_registry, key, comparable_args)
    if entry is not None:
        registry[key] = copy.deepcopy(entry)


def _replace_ssim_range(entry, stored_start, stored_count, incoming_start,
                        incoming_count, incoming_ssim):
    """Replace SSIM values for one stored range while preserving sparse legacy gaps."""
    replace_ids = {str(incoming_start + offset) for offset in range(incoming_count)}
    sparse_ssim = {
        str(stored_start + offset): float(value)
        for offset, value in enumerate(entry.get("best_ssim_list", []))
    }
    sparse_ssim.update({
        str(run_id): float(value)
        for run_id, value in entry.get("best_ssim_by_run_id", {}).items()
    })
    for run_id in replace_ids:
        sparse_ssim.pop(run_id, None)
    sparse_ssim.update({
        str(incoming_start + offset): value
        for offset, value in enumerate(incoming_ssim)
    })

    dense_ssim = [
        sparse_ssim.get(str(stored_start + offset))
        for offset in range(stored_count)
    ]
    if all(value is not None for value in dense_ssim):
        entry["best_ssim_list"] = dense_ssim
        entry.pop("best_ssim_by_run_id", None)
    else:
        entry.pop("best_ssim_list", None)
        entry["best_ssim_by_run_id"] = sparse_ssim


def _update_registry_entry(registry, key, comparable_args, best_psnr_list,
                           best_mse_list, best_ssim_list, label):
    """Store metrics for a sample range, appending or replacing contained ranges."""
    incoming = {
        "best_psnr_list": [float(v) for v in best_psnr_list],
        "best_mse_list": [float(v) for v in best_mse_list],
    }
    lengths = {len(values) for values in incoming.values()}
    if len(lengths) != 1:
        raise ValueError(f"{label} PSNR and MSE lists must have equal lengths.")

    incoming_start = int(comparable_args["run_id"])
    incoming_count = lengths.pop()
    if incoming_count != int(comparable_args["num_exp"]):
        raise ValueError(
            f"{label} received {incoming_count} metric values for "
            f"num_exp={comparable_args['num_exp']}."
        )
    incoming["best_ssim_list"] = [float(v) for v in best_ssim_list]
    if len(incoming["best_ssim_list"]) not in (0, incoming_count):
        raise ValueError(f"{label} SSIM list must be empty or match the PSNR and MSE lists.")

    stored_key, entry = find_registry_entry(registry, key, comparable_args)
    if entry is None:
        registry[key] = {
            "args": dict(comparable_args),
            **incoming,
        }
        return registry[key]

    if stored_key != key:
        entry = copy.deepcopy(entry)
        registry[key] = entry
        print(f"\nCopied legacy {label} registry entry to appendable key {key}.")

    entry_args = entry["args"]
    stored_start = int(entry_args["run_id"])
    stored_count = len(entry["best_psnr_list"])
    stored_ssim_list = entry.get("best_ssim_list", [])
    if len(stored_ssim_list) > stored_count:
        raise ValueError(f"Stored {label} SSIM list is longer than the PSNR list.")
    expected_start = stored_start + stored_count
    incoming_end = incoming_start + incoming_count
    if stored_start <= incoming_start and incoming_end <= expected_start:
        offset = incoming_start - stored_start
        entry["best_psnr_list"][offset:offset + incoming_count] = incoming["best_psnr_list"]
        entry["best_mse_list"][offset:offset + incoming_count] = incoming["best_mse_list"]
        _replace_ssim_range(
            entry, stored_start, stored_count, incoming_start, incoming_count,
            incoming["best_ssim_list"],
        )
        print(
            f"\nReplaced {incoming_count} existing {label} sample(s) for "
            f"run_id={incoming_start}..{incoming_end - 1}."
        )
        return entry
    if incoming_start != expected_start:
        raise ValueError(
            f"Cannot append {label} run_id={incoming_start}: stored samples cover "
            f"run_id={stored_start}..{expected_start - 1}, so the next run must use "
            f"--run_id {expected_start}."
        )

    entry["best_psnr_list"].extend(incoming["best_psnr_list"])
    entry["best_mse_list"].extend(incoming["best_mse_list"])
    if len(stored_ssim_list) == stored_count:
        entry.setdefault("best_ssim_list", []).extend(incoming["best_ssim_list"])
    elif incoming["best_ssim_list"]:
        sparse_ssim = entry.setdefault("best_ssim_by_run_id", {})
        sparse_ssim.update({
            str(incoming_start + offset): value
            for offset, value in enumerate(incoming["best_ssim_list"])
        })
    entry_args["num_exp"] = stored_count + incoming_count
    print(
        f"\nAppended {incoming_count} {label} sample(s); "
        f"registry entry now contains {entry_args['num_exp']} sample(s)."
    )
    return entry


def update_masked_registry(registry, key, comparable_args, best_psnr_list, best_mse_list, best_ssim_list):
    """Store or append a contiguous masked sample range for one configuration."""
    return _update_registry_entry(
        registry, key, comparable_args, best_psnr_list, best_mse_list, best_ssim_list,
        label="masked",
    )


def baseline_key_from_args(args):
    """Hash of baseline hyperparameters, excluding the sample range for appendable runs."""
    comparable = {
        "dataset": args.dataset,
        "network": args.network,
        "pretrained": bool(args.pretrained),
        "lr": args.lr,
        "gamma": args.gamma,
        "grad_loss": args.grad_loss,
        "num_dummy": args.num_dummy,
        "iteration": args.iteration,
        "tv_weight": args.tv_weight,
        "optimizer": args.optimizer,
        "num_restarts": args.num_restarts,
        "max_iteration": args.max_iteration,
        "history_size": args.history_size,
    }

    key_json = json.dumps(comparable, sort_keys=True)
    key_hash = hashlib.md5(key_json.encode("utf-8")).hexdigest()
    comparable["num_exp"] = args.num_exp
    comparable["run_id"] = args.run_id
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
    """Store or append a contiguous iDLG baseline sample range for one configuration."""
    return _update_registry_entry(
        registry, key, comparable_args, best_psnr_list, best_mse_list, best_ssim_list,
        label="iDLG baseline",
    )


def available_ssim_values(entry):
    """Return finite SSIM values from legacy lists and sparse future samples."""
    values = [float(v) for v in entry.get("best_ssim_list", [])]
    values.extend(float(v) for v in entry.get("best_ssim_by_run_id", {}).values())
    return [value for value in values if np.isfinite(value)]


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
        ssim_list = available_ssim_values(entry)
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
