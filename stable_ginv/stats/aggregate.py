"""NaN-safe aggregation helpers."""
import numpy as np


def mean_or_nan(values):
    """Return float mean, or nan for an empty sequence."""
    return float(np.mean(values)) if len(values) else float("nan")


def std_or_nan(values, ddof=1):
    """Return float std when there are enough samples for ddof, otherwise nan."""
    return float(np.std(values, ddof=ddof)) if len(values) > ddof else float("nan")


def median_or_nan(values):
    """Return float median, or nan for an empty sequence."""
    return float(np.median(values)) if len(values) else float("nan")
