"""Paired comparison statistics with lazy SciPy (HPC runtime may lack it)."""
import math
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
