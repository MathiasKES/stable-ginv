"""Shim: io_utils split into stable_ginv.{io,stats,registry} (Phase 4).

This module re-exports the public surface so existing callers keep working.
New code should import from stable_ginv.io / stable_ginv.stats / stable_ginv.registry.
"""
from stable_ginv.io import (
    safe_chmod,
    safe_makedirs,
    safe_write,
    safe_savefig,
    setstdout,
    StoragePaths,
    resolve_storage_paths,
    append_csv_rows,
    append_csv_row,
    parse_prefixes_with_fracs,
)
from stable_ginv.stats import (
    mean_or_nan,
    std_or_nan,
    median_or_nan,
    paired_t_ci,
    paired_summary,
    normality_str_from_ci,
    paired_metric_summaries,
)
from stable_ginv.registry import (
    masked_key_from_args,
    baseline_key_from_args,
    load_masked_registry,
    save_masked_registry,
    load_baseline_registry,
    save_baseline_registry,
    find_registry_entry,
    seed_registry_entry_from_fallback,
    update_masked_registry,
    update_idlg_baseline,
    available_ssim_values,
    write_baseline_summary_csv,
)

__all__ = [
    # io
    "safe_chmod",
    "safe_makedirs",
    "safe_write",
    "safe_savefig",
    "setstdout",
    "StoragePaths",
    "resolve_storage_paths",
    "append_csv_rows",
    "append_csv_row",
    "parse_prefixes_with_fracs",
    # stats
    "mean_or_nan",
    "std_or_nan",
    "median_or_nan",
    "paired_t_ci",
    "paired_summary",
    "normality_str_from_ci",
    "paired_metric_summaries",
    # registry
    "masked_key_from_args",
    "baseline_key_from_args",
    "load_masked_registry",
    "save_masked_registry",
    "load_baseline_registry",
    "save_baseline_registry",
    "find_registry_entry",
    "seed_registry_entry_from_fallback",
    "update_masked_registry",
    "update_idlg_baseline",
    "available_ssim_values",
    "write_baseline_summary_csv",
]
