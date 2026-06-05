# Phase 4 — `functions/io_utils.py` Teardown Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `functions/io_utils.py` (732 lines) into three focused packages — `stable_ginv/io/`, `stable_ginv/stats/`, `stable_ginv/registry/` — leaving `functions/io_utils.py` as a thin re-export shim, with no change to outputs, CLI, registry JSON, CSV content, or golden hashes.

**Architecture:** Filesystem/path/CSV-append helpers move to `stable_ginv/io/`; statistics (paired CIs, aggregation, lazy scipy) move to `stable_ginv/stats/`; registry key hashing, load/save, entry find/seed/update, and the baseline-summary CSV move to `stable_ginv/registry/`. The code is relocated verbatim (behavior-preserving). A new baseline-summary CSV golden locks `write_baseline_summary_csv` output **before** the registry logic moves. `functions/io_utils.py` becomes a re-export shim so all existing callers (`iDLG_mask.py`, `run_single_exp.py`, `manual_stats.py`, `show_img.py`, `functions/experiment_results.py`, `functions/jacobian_rank_sweep.py`, `helper/visualization.py`, `helper/plot_*.py`, and the test suite) keep working unchanged.

**Tech Stack:** Python 3, NumPy, lazy SciPy, pytest. Conda env: `stable-ginv`. All verification via `conda run -n stable-ginv`. Always `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"` before Python (DTU HPC C++ runtime requirement).

---

## Invariants (enforce every task)

- `conda run -n stable-ginv python -m pytest tests/ -q` stays green throughout.
- `conda run -n stable-ginv python iDLG_mask.py --help` works throughout.
- No change to reconstruction math, masking behavior, optimizer behavior, CLI
  defaults, registry key inputs, registry JSON structure, CSV
  filenames/columns/order/append behavior, or golden hashes.
- Code is moved **verbatim**. Do not "improve", reorder, or re-format the
  relocated function bodies — byte-identical behavior is the goal.
- `functions/io_utils.py` is a re-export shim by the end of the phase; every
  caller that still imports from it continues to work without edits.

## Module boundaries (current `functions/io_utils.py` → target)

| Target file | Functions moved (verbatim) |
|-------------|----------------------------|
| `stable_ginv/io/fs.py` | `safe_chmod`, `safe_makedirs`, `safe_write`, `safe_savefig`, `setstdout` |
| `stable_ginv/io/paths.py` | `StoragePaths` (new class) + `resolve_storage_paths` |
| `stable_ginv/io/csv.py` | `append_csv_rows`, `append_csv_row` |
| `stable_ginv/io/text.py` | `parse_prefixes_with_fracs` |
| `stable_ginv/stats/aggregate.py` | `mean_or_nan`, `std_or_nan`, `median_or_nan` |
| `stable_ginv/stats/paired.py` | `_load_scipy_stats`, `paired_t_ci`, `paired_summary`, `normality_str_from_ci`, `paired_metric_summaries` |
| `stable_ginv/registry/keys.py` | `_registry_key_hash`, `masked_key_from_args`, `baseline_key_from_args` |
| `stable_ginv/registry/store.py` | `_load_registry`, `_save_registry`, `load_masked_registry`, `save_masked_registry`, `load_baseline_registry`, `save_baseline_registry` |
| `stable_ginv/registry/entries.py` | `_config_without_sample_range`, `_entry_sample_range`, `_requested_sample_range`, `find_registry_entry`, `seed_registry_entry_from_fallback`, `_replace_ssim_range`, `_update_registry_entry`, `update_masked_registry`, `update_idlg_baseline`, `available_ssim_values` |
| `stable_ginv/registry/summary.py` | `write_baseline_summary_csv` |

`parse_prefixes_with_fracs` is a CLI argument-string parser with no registry/stats/IO
dependency; its natural long-term home is the future `stable_ginv/cli/` package (a later
phase). For Phase 4 it lives in `stable_ginv/io/text.py` so the shim stays a pure
re-export. Do not create a `cli/` package now — that is out of scope for this phase.

## File map

| Action | Path | Purpose |
|--------|------|---------|
| Create | `tests/golden/test_baseline_summary_csv_golden.py` | Locks `write_baseline_summary_csv` output before the registry move |
| Create | `tests/golden/fixtures/baseline_summary_csv_golden.json` | Golden fixture (generated, committed) |
| Create | `stable_ginv/io/__init__.py` | Public IO API |
| Create | `stable_ginv/io/fs.py` | `safe_*`, `setstdout` |
| Create | `stable_ginv/io/paths.py` | `StoragePaths`, `resolve_storage_paths` |
| Create | `stable_ginv/io/csv.py` | `append_csv_row(s)` |
| Create | `stable_ginv/io/text.py` | `parse_prefixes_with_fracs` |
| Create | `stable_ginv/stats/__init__.py` | Public stats API |
| Create | `stable_ginv/stats/aggregate.py` | `mean/std/median_or_nan` |
| Create | `stable_ginv/stats/paired.py` | paired CI + lazy scipy |
| Create | `stable_ginv/registry/__init__.py` | Public registry API |
| Create | `stable_ginv/registry/keys.py` | key hashing |
| Create | `stable_ginv/registry/store.py` | load/save registries |
| Create | `stable_ginv/registry/entries.py` | find/seed/update entries |
| Create | `stable_ginv/registry/summary.py` | baseline summary CSV |
| Modify | `functions/io_utils.py` | Replace with re-export shim |
| Modify | `tests/test_registry_keys.py` | Import from `stable_ginv.registry` |
| Modify | `tests/test_registry_append.py` | Import from `stable_ginv.registry` |
| Modify | `tests/test_safe_io.py` | Import from `stable_ginv.io` / `stable_ginv.registry` |
| Modify | `tests/golden/test_registry_key_golden.py` | Import from `stable_ginv.registry` |
| Modify | `tests/golden/test_baseline_summary_csv_golden.py` | Flip import to `stable_ginv.registry` |
| Modify | `docs/handover/HANDOVER_RESTRUCTURE.md` | Mark Phase 4 complete, update Next phase |

---

## Task 1: Lock the baseline-summary CSV with a golden (before any move)

**Files:**
- Create: `tests/golden/test_baseline_summary_csv_golden.py`
- Create: `tests/golden/fixtures/baseline_summary_csv_golden.json` (generated)

This golden locks the current `functions/io_utils.write_baseline_summary_csv` output
so the Phase 4 registry split is provably diff-zero. It imports from the **current**
home (`functions.io_utils`) so it is generated against pre-refactor behavior.

- [ ] **Step 1: Write the golden test**

Create `tests/golden/test_baseline_summary_csv_golden.py`:

```python
"""Golden test: baseline summary CSV is byte-stable across the Phase 4 registry split.

Guards write_baseline_summary_csv against accidental changes to columns, ordering,
float formatting, JSON-encoded list cells, or the dense/sparse SSIM handling.
"""
import os
import tempfile

from functions.io_utils import write_baseline_summary_csv
from tests.golden.helpers import load_or_regen


def _base_args(num_exp, run_id):
    return {
        "dataset": "cifar100",
        "network": "resnet18",
        "pretrained": False,
        "lr": 0.1,
        "gamma": 0.5,
        "grad_loss": "cos",
        "num_dummy": 1,
        "iteration": 5000,
        "num_exp": num_exp,
        "run_id": run_id,
        "tv_weight": 0.0,
        "optimizer": "signed_adamw",
        "num_restarts": 1,
        "max_iteration": 20,
        "history_size": 100,
    }


def _registry():
    # Three entries exercising: dense SSIM, sparse SSIM (best_ssim_by_run_id),
    # and an entry with no SSIM at all (empty list -> nan summary cells).
    return {
        "key_dense": {
            "args": _base_args(num_exp=3, run_id=0),
            "best_psnr_list": [10.5, 11.0, 9.5],
            "best_mse_list": [0.01, 0.02, 0.015],
            "best_ssim_list": [0.80, 0.85, 0.82],
        },
        "key_sparse": {
            "args": _base_args(num_exp=2, run_id=3),
            "best_psnr_list": [12.0, 13.0],
            "best_mse_list": [0.005, 0.004],
            "best_ssim_by_run_id": {"3": 0.90},
        },
        "key_nossim": {
            "args": _base_args(num_exp=1, run_id=7),
            "best_psnr_list": [8.0],
            "best_mse_list": [0.03],
        },
    }


def _produce():
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    try:
        write_baseline_summary_csv(path, _registry())
        with open(path) as f:
            return f.read()
    finally:
        os.remove(path)


def test_baseline_summary_csv_stable():
    golden = load_or_regen("baseline_summary_csv_golden.json", _produce)
    assert _produce() == golden
```

- [ ] **Step 2: Generate the fixture against current code**

Run:
```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
GOLDEN_REGEN=1 conda run -n stable-ginv python -m pytest tests/golden/test_baseline_summary_csv_golden.py -q
```
Expected: PASS; `tests/golden/fixtures/baseline_summary_csv_golden.json` is created.

- [ ] **Step 3: Run normally to confirm the golden asserts green**

Run:
```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -m pytest tests/golden/test_baseline_summary_csv_golden.py -q
```
Expected: 1 passed.

- [ ] **Step 4: Commit (`test:`)**

```bash
git add tests/golden/test_baseline_summary_csv_golden.py tests/golden/fixtures/baseline_summary_csv_golden.json
git commit -m "test: lock baseline summary CSV with a golden before Phase 4 registry split"
```

---

## Task 2: Create `stable_ginv/io/` package

**Files:**
- Create: `stable_ginv/io/fs.py`
- Create: `stable_ginv/io/paths.py`
- Create: `stable_ginv/io/csv.py`
- Create: `stable_ginv/io/text.py`
- Create: `stable_ginv/io/__init__.py`

All function bodies are copied **verbatim** from `functions/io_utils.py`.

- [ ] **Step 1: Create the package directory**

```bash
mkdir -p /home/mathias/GitHub/stable-ginv/stable_ginv/io
```

- [ ] **Step 2: Create `stable_ginv/io/fs.py`**

```python
"""Best-effort filesystem helpers: chmod/makedirs/write/savefig and stdout tee."""
import os
import sys
from datetime import datetime


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
```

- [ ] **Step 3: Create `stable_ginv/io/paths.py`**

`StoragePaths` is the OOP abstraction named in the spec; `resolve_storage_paths`
delegates to it and stays as the function all current callers use. Output strings
are byte-identical to the original.

```python
"""HPC/local storage path resolution."""
import os


class StoragePaths:
    """Resolve dataset/results paths: DTU HPC tree when writable, else local."""

    HPC_ROOT = "/work3/s234843/bachelor"

    @classmethod
    def resolve(cls, root_path="."):
        """Return (data_path, save_path) for DTU HPC when available, otherwise local paths."""
        if os.access(cls.HPC_ROOT, os.R_OK | os.W_OK | os.X_OK):
            return f"{cls.HPC_ROOT}/datasets", f"{cls.HPC_ROOT}/results"
        data_path = os.path.join(root_path, "data").replace("\\", "/")
        save_path = os.path.join(root_path, "results").replace("\\", "/")
        return data_path, save_path


def resolve_storage_paths(root_path="."):
    """Return (data_path, save_path) for DTU HPC when available, otherwise local paths."""
    return StoragePaths.resolve(root_path)
```

- [ ] **Step 4: Create `stable_ginv/io/csv.py`**

```python
"""Append-only CSV writers built on safe_write."""
import csv
import os

from stable_ginv.io.fs import safe_write


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
```

- [ ] **Step 5: Create `stable_ginv/io/text.py`**

```python
"""CLI argument-string parsing helpers."""


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
```

- [ ] **Step 6: Create `stable_ginv/io/__init__.py`**

```python
from stable_ginv.io.fs import (
    safe_chmod,
    safe_makedirs,
    safe_write,
    safe_savefig,
    setstdout,
)
from stable_ginv.io.paths import StoragePaths, resolve_storage_paths
from stable_ginv.io.csv import append_csv_rows, append_csv_row
from stable_ginv.io.text import parse_prefixes_with_fracs

__all__ = [
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
]
```

- [ ] **Step 7: Verify the package imports cleanly**

Run:
```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -c "import stable_ginv.io as io; print(io.__all__); print(io.resolve_storage_paths('.'))"
```
Expected: prints the `__all__` list and a `('./data', './results')`-style tuple (or the `/work3/...` pair on HPC). No error.

---

## Task 3: Create `stable_ginv/stats/` package

**Files:**
- Create: `stable_ginv/stats/aggregate.py`
- Create: `stable_ginv/stats/paired.py`
- Create: `stable_ginv/stats/__init__.py`

All function bodies copied **verbatim** from `functions/io_utils.py`.

- [ ] **Step 1: Create the package directory**

```bash
mkdir -p /home/mathias/GitHub/stable-ginv/stable_ginv/stats
```

- [ ] **Step 2: Create `stable_ginv/stats/aggregate.py`**

```python
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
```

- [ ] **Step 3: Create `stable_ginv/stats/paired.py`**

```python
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
```

- [ ] **Step 4: Create `stable_ginv/stats/__init__.py`**

```python
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
```

- [ ] **Step 5: Verify the package imports cleanly**

Run:
```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -c "import stable_ginv.stats as s; print(s.__all__); print(s.paired_t_ci([1,2,3,4],[1,1,1,1])['mean_diff'])"
```
Expected: prints `__all__` and `1.5`. No error.

---

## Task 4: Create `stable_ginv/registry/` package

**Files:**
- Create: `stable_ginv/registry/keys.py`
- Create: `stable_ginv/registry/store.py`
- Create: `stable_ginv/registry/entries.py`
- Create: `stable_ginv/registry/summary.py`
- Create: `stable_ginv/registry/__init__.py`

All function bodies copied **verbatim** from `functions/io_utils.py`. Do not alter the
comparable-args dicts in `keys.py` (registry-key inputs are an invariant).

- [ ] **Step 1: Create the package directory**

```bash
mkdir -p /home/mathias/GitHub/stable-ginv/stable_ginv/registry
```

- [ ] **Step 2: Create `stable_ginv/registry/keys.py`**

```python
"""Registry key hashing from masking/baseline hyperparameters."""
import hashlib
import json


def _registry_key_hash(comparable):
    """MD5 of the comparable-args dict with stable key ordering.

    Call before num_exp/run_id are added so appendable runs share one key.
    """
    key_json = json.dumps(comparable, sort_keys=True)
    return hashlib.md5(key_json.encode("utf-8")).hexdigest()


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
        # Regime marker: the last FC layer is no longer force-included in the
        # reconstruction mask (it is used only for label inference). This field
        # gives new runs a distinct key so their results never collide with or
        # append to pre-change entries that were produced with FC force-included.
        "fc_forced": False,
    }
    key_hash = _registry_key_hash(comparable)
    comparable["num_exp"] = args.num_exp
    comparable["run_id"] = args.run_id
    return key_hash, comparable


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

    key_hash = _registry_key_hash(comparable)
    comparable["num_exp"] = args.num_exp
    comparable["run_id"] = args.run_id
    return key_hash, comparable
```

- [ ] **Step 3: Create `stable_ginv/registry/store.py`**

```python
"""JSON registry load/save for masked and baseline registries."""
import json
import os

from stable_ginv.io import safe_write


def _load_registry(path):
    """Load a JSON registry from path; return an empty dict if it does not exist."""
    if not os.path.isfile(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def load_masked_registry(path):
    """Load JSON masked registry from path; returns empty dict if file does not exist."""
    return _load_registry(path)


def _save_registry(path, registry):
    """Write a JSON registry to path. Failures are logged, not raised."""
    return safe_write(path, lambda f: json.dump(registry, f, indent=2))


def save_masked_registry(path, registry):
    """Write masked registry dict to JSON at path. Failures are logged, not raised."""
    return _save_registry(path, registry)


def load_baseline_registry(path):
    """Load JSON baseline registry from path; returns empty dict if file does not exist."""
    return _load_registry(path)


def save_baseline_registry(path, registry):
    """Write baseline registry dict to JSON at path. Failures are logged, not raised."""
    return _save_registry(path, registry)
```

- [ ] **Step 4: Create `stable_ginv/registry/entries.py`**

```python
"""Find, seed, and update registry entries for contiguous sample ranges."""
import copy

import numpy as np


def _config_without_sample_range(args):
    return {
        name: value
        for name, value in args.items()
        if name not in ("num_exp", "run_id")
    }


def _entry_sample_range(entry):
    args = entry.get("args", {})
    if "run_id" not in args:
        return None
    start = int(args["run_id"])
    count = len(entry.get("best_psnr_list", []))
    return start, start + count


def _requested_sample_range(comparable_args):
    if "run_id" not in comparable_args or "num_exp" not in comparable_args:
        return None
    start = int(comparable_args["run_id"])
    return start, start + int(comparable_args["num_exp"])


def find_registry_entry(registry, key, comparable_args, allow_append_predecessor=False):
    """Return the best current or legacy registry entry matching one configuration."""
    target_config = _config_without_sample_range(comparable_args)
    matches = [
        (stored_key, entry)
        for stored_key, entry in registry.items()
        if _config_without_sample_range(entry.get("args", {})) == target_config
    ]
    requested_range = _requested_sample_range(comparable_args)
    if requested_range is None:
        if key in registry:
            return key, registry[key]
        if len(matches) <= 1:
            return matches[0] if matches else (None, None)
        raise ValueError(
            "Multiple legacy registry entries match this configuration. "
            "Provide a sample range or select the registry entry explicitly."
        )

    requested_start, requested_end = requested_range
    containing = [
        (stored_key, entry, start, end)
        for stored_key, entry in matches
        if (sample_range := _entry_sample_range(entry)) is not None
        for start, end in [sample_range]
        if start <= requested_start and requested_end <= end
    ]
    if containing:
        stored_key, entry, _, _ = min(
            containing,
            key=lambda item: (
                item[3] - item[2],
                item[2],
                item[0],
            ),
        )
        return stored_key, entry

    if allow_append_predecessor:
        predecessors = [
            (stored_key, entry, start, end)
            for stored_key, entry in matches
            if (sample_range := _entry_sample_range(entry)) is not None
            for start, end in [sample_range]
            if end == requested_start
        ]
        if predecessors:
            stored_key, entry, _, _ = max(
                predecessors,
                key=lambda item: (
                    item[3] - item[2],
                    -item[2],
                    item[0],
                ),
            )
            return stored_key, entry

    return None, None


def seed_registry_entry_from_fallback(registry, fallback_registry, key, comparable_args):
    """Copy one matching read-only fallback entry into a writable registry."""
    if key in registry or find_registry_entry(registry, key, comparable_args)[1] is not None:
        return
    _, entry = find_registry_entry(
        fallback_registry, key, comparable_args, allow_append_predecessor=True
    )
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
    """Store metrics for a sample range, appending unseen suffixes or replacing contained ranges."""
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

    if key in registry:
        stored_key, entry = key, registry[key]
    else:
        stored_key, entry = find_registry_entry(
            registry, key, comparable_args, allow_append_predecessor=True
        )
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
    if stored_start <= incoming_start < expected_start < incoming_end:
        overlap_count = expected_start - incoming_start
        incoming_start = expected_start
        incoming_count -= overlap_count
        incoming = {
            name: values[overlap_count:]
            for name, values in incoming.items()
        }
        print(
            f"\nSkipped {overlap_count} existing {label} sample(s) and will append "
            f"run_id={incoming_start}..{incoming_end - 1}."
        )
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
```

- [ ] **Step 5: Create `stable_ginv/registry/summary.py`**

```python
"""Baseline registry summary CSV writer."""
import csv
import json

import numpy as np

from stable_ginv.io import safe_write
from stable_ginv.registry.entries import available_ssim_values


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
```

- [ ] **Step 6: Create `stable_ginv/registry/__init__.py`**

```python
from stable_ginv.registry.keys import (
    masked_key_from_args,
    baseline_key_from_args,
)
from stable_ginv.registry.store import (
    load_masked_registry,
    save_masked_registry,
    load_baseline_registry,
    save_baseline_registry,
)
from stable_ginv.registry.entries import (
    find_registry_entry,
    seed_registry_entry_from_fallback,
    update_masked_registry,
    update_idlg_baseline,
    available_ssim_values,
)
from stable_ginv.registry.summary import write_baseline_summary_csv

__all__ = [
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
```

- [ ] **Step 7: Verify the package imports cleanly and keys are unchanged**

Run:
```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -c "
from types import SimpleNamespace
from stable_ginv.registry import masked_key_from_args
from functions.io_utils import masked_key_from_args as old
a = SimpleNamespace(dataset='cifar100', network='vgg13', pretrained=False, lr=0.1, gamma=0.5, grad_loss='cos', num_dummy=1, iteration=5000, tv_weight=0.0, optimizer='signed_adamw', num_restarts=1, max_iteration=20, history_size=100, mask_mode='gradsize_topk', gradsize_topk=None, gradsize_topfrac=0.1, gradsize_metric='abs', prefixes='conv1:0.5,fc', num_exp=30, run_id=0)
assert masked_key_from_args(a)[0] == old(a)[0], 'KEY DRIFT'
print('registry key parity OK:', masked_key_from_args(a)[0])
"
```
Expected: prints `registry key parity OK: <hash>`. No assertion error. (At this point `functions.io_utils` is still the original module, so this proves the moved `keys.py` matches it byte-for-byte.)

---

## Task 5: Convert `functions/io_utils.py` to a re-export shim

**Files:**
- Modify: `functions/io_utils.py`

The shim must re-export every public symbol that any caller imports. Verified callers
and their imports:
- `iDLG_mask.py`: `baseline_key_from_args`, `load_baseline_registry`, `save_baseline_registry`, `update_idlg_baseline`, `write_baseline_summary_csv`, `parse_prefixes_with_fracs`, `masked_key_from_args`, `load_masked_registry`, `save_masked_registry`, `update_masked_registry`, `append_csv_rows`, `find_registry_entry`, `resolve_storage_paths`, `safe_makedirs`, `seed_registry_entry_from_fallback`, `setstdout`
- `run_single_exp.py`: `setstdout`
- `manual_stats.py`: `resolve_storage_paths`, `safe_chmod`, `safe_makedirs`
- `show_img.py`: `resolve_storage_paths`, `safe_makedirs`, `safe_savefig`
- `functions/experiment_results.py`: `append_csv_row`, `mean_or_nan`, `median_or_nan`, `paired_metric_summaries`, `std_or_nan`
- `functions/jacobian_rank_sweep.py`: `parse_prefixes_with_fracs`, `resolve_storage_paths`, `safe_makedirs`, `safe_savefig`
- `functions/rank_reconstruction_plot.py`: `resolve_storage_paths`
- `helper/visualization.py`: `normality_str_from_ci`, `paired_t_ci`, `safe_savefig`, `safe_chmod`, `safe_makedirs`, `safe_write`
- `helper/plot_model_parameter_counts.py`: `safe_savefig`
- `helper/plot_combined_masking_sweep_summary.py`: `safe_makedirs`, `safe_savefig`, `safe_write`
- `helper/plot_paired_masking_violin.py`: `resolve_storage_paths`, `safe_makedirs`, `safe_savefig`, `safe_write`
- `helper/plot_masking_sweep_csv.py`: `find_registry_entry`, `safe_makedirs`, `safe_savefig`, `safe_write`
- `artifacts/figures/precompute_lfw_landscape.py`: `setstdout`
- Tests: `safe_chmod`, `safe_makedirs`, `safe_savefig`, `safe_write`, `save_baseline_registry`, `save_masked_registry`, `write_baseline_summary_csv`, `baseline_key_from_args`, `masked_key_from_args`, `find_registry_entry`, `seed_registry_entry_from_fallback`, `update_idlg_baseline`, `update_masked_registry`

`available_ssim_values` and `StoragePaths` were public in the original module and are
re-exported too, to preserve the exact public surface.

- [ ] **Step 1: Replace the entire content of `functions/io_utils.py`**

```python
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
```

- [ ] **Step 2: Run the full test suite (everything still imports from the shim)**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -m pytest tests/ -q
```
Expected: all tests PASS, including `tests/golden/` (the baseline-summary CSV golden
now flows through the shim → identical bytes). If **any** golden hash or the CSV golden
differs, STOP and debug — do not regenerate fixtures to force a pass.

- [ ] **Step 3: Confirm the CLI still works**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python iDLG_mask.py --help
```
Expected: help text printed, exit 0.

- [ ] **Step 4: Lint the new package and the shim**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -m pyflakes stable_ginv functions/io_utils.py
```
Expected: no output (clean). If pyflakes is unavailable, use
`conda run -n stable-ginv python -m py_compile stable_ginv/io/*.py stable_ginv/stats/*.py stable_ginv/registry/*.py functions/io_utils.py` (expect no output).

- [ ] **Step 5: Commit (`refactor:`)**

```bash
git add stable_ginv/io/ stable_ginv/stats/ stable_ginv/registry/ functions/io_utils.py
git commit -m "refactor: split functions/io_utils.py into stable_ginv/{io,stats,registry}"
```

---

## Task 6: Point tests and goldens at the canonical import paths

**Files:**
- Modify: `tests/test_registry_keys.py`
- Modify: `tests/test_registry_append.py`
- Modify: `tests/test_safe_io.py`
- Modify: `tests/golden/test_registry_key_golden.py`
- Modify: `tests/golden/test_baseline_summary_csv_golden.py`

These updates make the new dependency explicit (no longer routed through the shim).
The shim keeps non-test callers in `functions/` and `helper/` working — those are
migrated in later phases.

- [ ] **Step 1: Update `tests/test_registry_keys.py`**

Find:
```python
from functions.io_utils import baseline_key_from_args, masked_key_from_args
```
Replace with:
```python
from stable_ginv.registry import baseline_key_from_args, masked_key_from_args
```

- [ ] **Step 2: Update `tests/test_registry_append.py`**

Find:
```python
from functions.io_utils import (
    baseline_key_from_args,
    find_registry_entry,
    masked_key_from_args,
    seed_registry_entry_from_fallback,
    update_idlg_baseline,
    update_masked_registry,
)
```
Replace with:
```python
from stable_ginv.registry import (
    baseline_key_from_args,
    find_registry_entry,
    masked_key_from_args,
    seed_registry_entry_from_fallback,
    update_idlg_baseline,
    update_masked_registry,
)
```

- [ ] **Step 3: Update `tests/test_safe_io.py`**

Find:
```python
from functions.io_utils import (
    safe_chmod,
    safe_makedirs,
    safe_savefig,
    safe_write,
    save_baseline_registry,
    save_masked_registry,
    write_baseline_summary_csv,
)
```
Replace with:
```python
from stable_ginv.io import (
    safe_chmod,
    safe_makedirs,
    safe_savefig,
    safe_write,
)
from stable_ginv.registry import (
    save_baseline_registry,
    save_masked_registry,
    write_baseline_summary_csv,
)
```

Also update the module docstring's first line if it names the old module. Find:
```python
"""Tests for the safe-write helpers in functions.io_utils.
```
Replace with:
```python
"""Tests for the safe-write helpers in stable_ginv.io.
```

- [ ] **Step 4: Update `tests/golden/test_registry_key_golden.py`**

Find:
```python
from functions.io_utils import masked_key_from_args
```
Replace with:
```python
from stable_ginv.registry import masked_key_from_args
```

Also update the docstring reference. Find:
```python
Guards functions/io_utils.masked_key_from_args against accidental changes to the
```
Replace with:
```python
Guards stable_ginv.registry.masked_key_from_args against accidental changes to the
```

- [ ] **Step 5: Update `tests/golden/test_baseline_summary_csv_golden.py`**

Find:
```python
from functions.io_utils import write_baseline_summary_csv
```
Replace with:
```python
from stable_ginv.registry import write_baseline_summary_csv
```

- [ ] **Step 6: Run the full suite — golden hashes must be unchanged**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -m pytest tests/ -q
```
Expected: all tests PASS. The baseline-summary CSV golden and registry-key golden
read identical fixtures (the underlying code is byte-identical via verbatim move). If
any differ, STOP and debug.

- [ ] **Step 7: Commit (`test:`)**

```bash
git add tests/test_registry_keys.py tests/test_registry_append.py tests/test_safe_io.py tests/golden/test_registry_key_golden.py tests/golden/test_baseline_summary_csv_golden.py
git commit -m "test: import registry/io/stats helpers from stable_ginv canonical paths"
```

---

## Task 7: Update the handover and finalize

**Files:**
- Modify: `docs/handover/HANDOVER_RESTRUCTURE.md`

- [ ] **Step 1: Mark Phase 4 complete in the Status section**

Find:
```markdown
- [ ] Phase 4 — `io_utils` teardown → `registry/` + `stats/` + `io/`.
```
Replace with:
```markdown
- [x] Phase 4 — `io_utils` teardown → `registry/` + `stats/` + `io/`.
```

- [ ] **Step 2: Replace the `## Next phase` section**

Find the entire `## Next phase` block and replace it with:
```markdown
## Next phase

Write the Phase 5 plan from the spec, then implement. Phase 5 extracts
`stable_ginv/recon/` — the `ReconstructionRunner` optimization loop, `EarlyStopPolicy`,
`LabelInference` (one-time, from the original unmasked FC gradient), and the scheduler
(`helper/training_utils.py`). `run_single_exp.py` must stay importable at its current
path as a worker shim that delegates to `stable_ginv.recon`, so multiprocessing spawn
keeps working. The recon-worker golden (`tests/golden/test_recon_worker_golden.py`)
guards the numerics — do not let it drift. Keep this file's Status section current at
every phase boundary.
```

- [ ] **Step 3: Update the CSV-row golden note**

In the "Golden harness (Phase 0)" section, find:
```markdown
a full-CLI run would append to real experiment data. CSV-row goldens are added in
Phase 6, just before the `experiment_results` refactor.
```
Replace with:
```markdown
a full-CLI run would append to real experiment data. The baseline-summary CSV golden
(`tests/golden/test_baseline_summary_csv_golden.py`) was added in Phase 4 to guard the
registry-summary writer; the per-run `experiment_results` CSV-row golden is added in
Phase 6, just before that refactor.
```

- [ ] **Step 4: Add the baseline-summary CSV golden to the golden-harness list**

In the "Golden harness (Phase 0)" section, find:
```markdown
- `tests/golden/test_recon_worker_golden.py` — in-process recon numerics (CPU).
```
Replace with:
```markdown
- `tests/golden/test_recon_worker_golden.py` — in-process recon numerics (CPU).
- `tests/golden/test_baseline_summary_csv_golden.py` — baseline-summary CSV bytes
  (added Phase 4, guards the registry-summary writer).
```

- [ ] **Step 5: Final full verification**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -m pytest tests/ -q
conda run -n stable-ginv python iDLG_mask.py --help
git diff --check
```
Expected: all tests PASS, help prints, no whitespace errors.

- [ ] **Step 6: Commit (`docs:`)**

```bash
git add docs/handover/HANDOVER_RESTRUCTURE.md
git commit -m "docs: mark Phase 4 complete in HANDOVER_RESTRUCTURE"
```

---

## Self-review checklist

**Spec coverage (Section 1 mapping `functions/io_utils.py` → `registry/` + `stats/` + `io/`):**
- [x] `io/` holds `StoragePaths`, CSV writers, `safe_makedirs`/`safe_savefig` (+ `safe_chmod`/`safe_write`/`setstdout`/`parse_prefixes_with_fracs`)
- [x] `stats/` holds paired comparisons, confidence intervals, lazy scipy, aggregation
- [x] `registry/` holds key hashing, load/save, find/seed/update, baseline summary CSV
- [x] `StoragePaths` OOP abstraction created (Section 1 + key abstractions list)
- [x] Registry comparable-args dicts (incl. `fc_forced` marker) moved verbatim — keys unchanged
- [x] CSV-row golden added before moving registry logic (baseline-summary CSV golden, Task 1)
- [x] `functions/io_utils.py` is a shim; all callers unmodified
- [x] Categorized commits: `test:` (golden) / `refactor:` (split) / `test:` (imports) / `docs:`
- [x] HANDOVER_RESTRUCTURE.md Status + Next phase updated

**Placeholder scan:** no TBD/TODO/"handle edge cases"/"similar to" — every module body
is full verbatim code.

**Type/name consistency:**
- [x] `StoragePaths.resolve(root_path=".")` returns the same `(data_path, save_path)` tuple `resolve_storage_paths` returns; `resolve_storage_paths` delegates to it.
- [x] `write_baseline_summary_csv` imports `available_ssim_values` from `stable_ginv.registry.entries` and `safe_write` from `stable_ginv.io` — both defined in earlier tasks.
- [x] `store.py` and `summary.py` import `safe_write` from `stable_ginv.io` (defined Task 2 before Task 4).
- [x] Shim `__all__` lists exactly the union of the three packages' public names plus `StoragePaths`/`available_ssim_values`.

**Potential issues to watch:**
- No circular imports: `io` imports nothing from `stats`/`registry`; `registry` imports `io`; `stats` is independent; the shim imports all three. Import order in the shim is `io` → `stats` → `registry`, matching the dependency direction.
- The baseline-summary CSV golden fixture must be generated in Task 1 against `functions.io_utils` (pre-move) so it captures pre-refactor bytes; Tasks 5–6 must not change those bytes.
- `parse_prefixes_with_fracs` intentionally lives in `io/text.py` for this phase (no `cli/` package yet) — do not create `stable_ginv/cli/` in Phase 4.
- Do not add the new subpackages to `pyproject.toml` `[tool.setuptools] packages`: the editable install puts the repo root on `sys.path`, so `stable_ginv.io`/`.stats`/`.registry` import as on-disk subpackages exactly like `stable_ginv.masking`/`.metrics` already do.
```