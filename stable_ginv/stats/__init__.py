"""Statistics: NaN-safe aggregates and paired comparisons with confidence intervals."""
from stable_ginv.stats.aggregate import mean_or_nan, std_or_nan, median_or_nan
from stable_ginv.stats.paired import (
    paired_t_ci,
    paired_summary,
    normality_str_from_ci,
    paired_metric_summaries,
)

__all__ = [
    "mean_or_nan",
    "std_or_nan",
    "median_or_nan",
    "paired_t_ci",
    "paired_summary",
    "normality_str_from_ci",
    "paired_metric_summaries",
]
