# Phase 6 — `stable_ginv/experiment/` + `stable_ginv/cli/batch.py` Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the batch orchestration (`iDLG_mask.py`) and the result-aggregation/CSV layer (`functions/experiment_results.py`) into a `stable_ginv/experiment/` package (`BatchExperimentRunner`, `ResultAggregator`, `RestartSelector`, `results.py`) plus a `stable_ginv/cli/batch.py` `main()`, leaving `iDLG_mask.py` as a thin `from stable_ginv.cli.batch import main` wrapper and `functions/experiment_results.py` as a re-export shim, with **zero** change to scheduling behavior, CSV columns/order/rounding, registry JSON, stdout, or CLI flags/defaults.

**Architecture:** Mirrors the Phase 5 recon split (thin OOP over verbatim relocated code). The aggregation/CSV functions move **verbatim** into `stable_ginv/experiment/results.py` (only the `functions.io_utils` import line is repointed to canonical `stable_ginv.stats`/`stable_ginv.io`), gaining two thin façades — `ResultAggregator` (delegates to `create_metric_accumulators`/`append_result_metrics`/`compute_aggregate_stats`) and `RestartSelector` (delegates to `merge_restart_results`). The `iDLG_mask.py` scheduling section moves **verbatim** into `BatchExperimentRunner.run(handle_result)` (GPU detection, multiprocessing spawn, round-robin restart interleaving, restart merging, abort-on-worker-failure), and the rest of `main()` moves verbatim into `stable_ginv/cli/batch.py` with four mechanical `metric_accumulators → aggregator` substitutions. Both old files become shims so every existing caller (`manual_stats.py`, the test suite, HPC `python iDLG_mask.py` commands, the multiprocessing spawn target) keeps working unedited.

**Tech Stack:** Python 3, PyTorch (multiprocessing spawn), NumPy, tqdm, pytest. Conda env: `stable-ginv`. All verification via `conda run -n stable-ginv`. Always `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"` before Python (DTU HPC C++ runtime requirement).

---

## Invariants (enforce every task)

- `conda run -n stable-ginv python -m pytest tests/ -q` stays green throughout, including all goldens and the **new** `tests/golden/test_experiment_results_csv_golden.py`.
- `conda run -n stable-ginv python iDLG_mask.py --help` works throughout (byte-identical help text; the parser is untouched).
- No change to: worker scheduling/output ordering, abort-on-worker-failure behavior, restart merging, metric accumulation, aggregate statistics, paired-statistics math, registry key inputs / JSON, CSV filenames/columns/order/rounding/append behavior, stdout lines, plot filenames, or CLI flags/defaults.
- Moved bodies are relocated **verbatim**. The only permitted edits are the documented ones in each task (import repointing, the closure→method/callback restructure, the `metric_accumulators → aggregator` substitutions, and dropping the two now-hoisted panel-flush calls from the scheduling branches). Do not reorder, reformat, or "improve" the relocated code, and do not alter any string literal inside a `print(...)`.
- **Never regenerate a golden to make a refactor pass.** If `test_experiment_results_csv_golden.py` (or any golden) drifts, STOP and debug — the move was not verbatim. Do not edit any fixture.
- `iDLG_mask.py` ends the phase as a thin wrapper exposing `main`; `functions/experiment_results.py` ends as a re-export shim. Every existing caller keeps working unedited.
- The multiprocessing spawn target stays `stable_ginv.recon.runner.run_single_experiment` (unchanged from Phase 5); `BatchExperimentRunner` imports it canonically from `stable_ginv.recon`.

## Module boundaries (current → target)

| Target file | Content |
|-------------|---------|
| `stable_ginv/experiment/results.py` | Verbatim `functions/experiment_results.py` (imports repointed) + `ResultAggregator` + `RestartSelector` façades |
| `stable_ginv/experiment/runner.py` | `BatchExperimentRunner` (scheduling/multiprocessing/abort) + `ExperimentRunAborted` |
| `stable_ginv/experiment/__init__.py` | Public experiment API |
| `stable_ginv/cli/batch.py` | `main()` — arg parse → `ExperimentConfig` → data load → `BatchExperimentRunner` → registry/paired/CSV/summary |
| `stable_ginv/cli/__init__.py` | Package marker (no side-effect imports) |
| `iDLG_mask.py` | Thin wrapper → `from stable_ginv.cli.batch import main; main()` |
| `functions/experiment_results.py` | Re-export shim |

### Imports kept as-is (their canonical home is a later phase — out of scope here)

- `from functions.idlg_cli import parse_idlg_args` — **keep**; `idlg_cli.py` → `stable_ginv/cli/args.py` is deferred to a later phase. Keeping it makes `--help` trivially byte-identical.
- `from functions.Dataset import load_dataset` — **keep**; `Dataset.py` → `stable_ginv/data/` is a later phase.
- `from helper.visualization import (...)` — **keep** (loaded lazily inside `_load_visualization_helpers`); `helper/visualization.py` → `stable_ginv/viz/` is Phase 7.

### Canonical imports inside the new modules

- `stable_ginv/experiment/results.py`: `from stable_ginv.io import append_csv_row` and `from stable_ginv.stats import mean_or_nan, median_or_nan, paired_metric_summaries, std_or_nan` (Phase 4 canonical homes).
- `stable_ginv/experiment/runner.py`: `from stable_ginv.recon import run_single_experiment` (Phase 5), `from stable_ginv.experiment.results import RestartSelector`.
- `stable_ginv/cli/batch.py`: `from stable_ginv.io import ...`, `from stable_ginv.registry import ...`, `from stable_ginv.experiment.results import ...`, `from stable_ginv.experiment.runner import BatchExperimentRunner`, `from stable_ginv.config import ExperimentConfig`.

## File map

| Action | Path | Purpose |
|--------|------|---------|
| Create | `tests/golden/test_experiment_results_csv_golden.py` | Locks per-run CSV rows (columns/order/rounding/env fields) before the move |
| Create | `tests/golden/fixtures/experiment_results_csv_golden.json` | Golden fixture (auto-generated on first run) |
| Create | `stable_ginv/experiment/__init__.py` | Public experiment API |
| Move   | `functions/experiment_results.py` → `stable_ginv/experiment/results.py` | Verbatim relocation (imports repointed) + façades |
| Create | `functions/experiment_results.py` | Re-export shim (new file at old path) |
| Create | `stable_ginv/experiment/runner.py` | `BatchExperimentRunner` + `ExperimentRunAborted` |
| Create | `stable_ginv/cli/__init__.py` | Package marker |
| Create | `stable_ginv/cli/batch.py` | `main()` orchestration |
| Modify | `iDLG_mask.py` | Replace with thin wrapper |
| Modify | `tests/golden/test_experiment_results_csv_golden.py` | Repoint import to `stable_ginv.experiment.results` |
| Modify | `docs/handover/HANDOVER_RESTRUCTURE.md` | Mark Phase 6 complete, update Next phase |

**Do not edit `pyproject.toml`.** The editable install puts the repo root on `sys.path`, so `stable_ginv.experiment` and `stable_ginv.cli` import as on-disk subpackages exactly like `stable_ginv.recon`/`.io`/`.registry` already do.

---

## Task 1: Lock the per-run experiment_results CSV rows with a golden (before the move)

**Files:**
- Create: `tests/golden/test_experiment_results_csv_golden.py`
- Create (auto): `tests/golden/fixtures/experiment_results_csv_golden.json`

The spec defers this golden to "just before this refactor" (Section 2.3). It captures `build_common_csv_fields` + `build_exp_result_rows` + `append_csv_rows` end-to-end into CSV bytes, locking column names, column order, the iDLG/masked row construction, float rounding, and the `job_id`/`device`/`Run by` environment fields. `methods="both"` exercises both output rows. It imports from the **current** `functions.experiment_results` path; Task 3 repoints it after the move.

- [ ] **Step 1: Write the test**

Create `tests/golden/test_experiment_results_csv_golden.py`:

```python
"""Golden test: per-run experiment_results CSV rows are byte-stable across the Phase 6 split.

Locks build_common_csv_fields + build_exp_result_rows + append_csv_rows end-to-end:
column names, column order, the iDLG/masked row construction, float rounding, and the
job_id/device/Run-by environment fields. methods='both' exercises both output rows.
The env vars and timestamp/argv are fixed so the bytes are deterministic.
"""
import os
import tempfile

from functions.experiment_results import (
    build_common_csv_fields,
    build_exp_result_rows,
)
from functions.io_utils import append_csv_rows
from tests.golden.helpers import load_or_regen


# Only the keys build_exp_result_rows actually reads from `stats`.
_STATS = {
    "med_best_loss_idlg": 0.123456789,
    "avg_best_loss_idlg": 0.234567891,
    "med_best_mse_idlg": 0.000123456789,
    "avg_best_mse_idlg": 0.000234567891,
    "avg_best_psnr_idlg": 21.111111,
    "std_best_psnr_idlg": 1.222222,
    "avg_best_ssim_idlg": 0.811111,
    "std_best_ssim_idlg": 0.022222,
    "med_best_loss_masked": 0.323456789,
    "avg_best_loss_masked": 0.434567891,
    "med_best_mse_masked": 0.000323456789,
    "avg_best_mse_masked": 0.000434567891,
    "avg_best_psnr_masked": 18.999999,
    "std_best_psnr_masked": 2.333333,
    "avg_best_ssim_masked": 0.733333,
    "std_best_ssim_masked": 0.044444,
}

# Only the *_str keys build_exp_result_rows reads from the paired report.
_PAIRED = {
    "mse_ci_str": "[-0.0002, 0.0004]",
    "psnr_ci_str": "[-3.0, -1.2]",
    "ssim_ci_str": "[-0.10, -0.06]",
    "mse_significant_str": "yes",
    "psnr_significant_str": "yes",
    "ssim_significant_str": "no",
    "psnr_normality_str": "normal",
    "mse_normality_str": "normal",
    "ssim_normality_str": "non-normal",
}


def _produce():
    env_keys = ("LSB_JOBID", "LSB_INTERACTIVE", "USER", "LSB_QUEUE")
    saved = {k: os.environ.get(k) for k in env_keys}
    os.environ["LSB_JOBID"] = "123456"
    os.environ.pop("LSB_INTERACTIVE", None)
    os.environ["USER"] = "golden_user"
    os.environ["LSB_QUEUE"] = "gpuv100"
    try:
        common = build_common_csv_fields(
            "20260101_000000",
            ["iDLG_mask.py", "--methods", "both", "--network", "LeNet"],
            "MNIST", "LeNet", False,
            1, 1.0, 0.5, 10, 2, 0.0, "lbfgs", 20, 100,
        )
        rows, fieldnames = build_exp_result_rows(
            "both", common, _STATS, _PAIRED,
            "baselinekey123", "maskedkey456",
            "gradsize_topfrac", "conv1:1.0,fc:1.0", 0.5, "panelA.png|panelB.png",
        )
        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        try:
            append_csv_rows(path, rows, fieldnames)
            with open(path) as f:
                return f.read()
        finally:
            os.remove(path)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_experiment_results_csv_rows_stable():
    golden = load_or_regen("experiment_results_csv_golden.json", _produce)
    assert _produce() == golden
```

- [ ] **Step 2: Generate the fixture and verify it passes against current code**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -m pytest tests/golden/test_experiment_results_csv_golden.py -q
```
Expected: 1 passed. `load_or_regen` writes `tests/golden/fixtures/experiment_results_csv_golden.json` on first run (it does not yet exist), then the assert compares `_produce()` against it. Confirm the fixture file now exists:

```bash
ls tests/golden/fixtures/experiment_results_csv_golden.json
```
Expected: the path prints (file created).

- [ ] **Step 3: Commit (`test:`)**

```bash
git add tests/golden/test_experiment_results_csv_golden.py tests/golden/fixtures/experiment_results_csv_golden.json
git commit -m "test: lock experiment_results CSV rows before Phase 6 orchestration split"
```

---

## Task 2: Move `experiment_results.py` → `stable_ginv/experiment/results.py` (verbatim) + add façades + shim

**Files:**
- Create: `stable_ginv/experiment/__init__.py`
- Move: `functions/experiment_results.py` → `stable_ginv/experiment/results.py`
- Create: `functions/experiment_results.py` (re-export shim)

- [ ] **Step 1: Create the package directory and move the file (preserving history)**

```bash
mkdir -p /home/mathias/GitHub/stable-ginv/stable_ginv/experiment
git mv functions/experiment_results.py stable_ginv/experiment/results.py
```

- [ ] **Step 2: Repoint the imports in `stable_ginv/experiment/results.py`**

This is the **only** edit to the relocated body. Find (top of file):
```python
from functions.io_utils import (
    append_csv_row,
    mean_or_nan,
    median_or_nan,
    paired_metric_summaries,
    std_or_nan,
)
```
Replace with:
```python
from stable_ginv.io import append_csv_row
from stable_ginv.stats import (
    mean_or_nan,
    median_or_nan,
    paired_metric_summaries,
    std_or_nan,
)
```
Leave the other imports (`import hashlib`, `import json`, `import os`, `import numpy as np`) and every function/constant body unchanged.

- [ ] **Step 3: Append the two thin façades to `stable_ginv/experiment/results.py`**

Add at the **end** of the file (they delegate to the verbatim module-level functions defined above, so behavior is identical by construction):

```python
class ResultAggregator:
    """Accumulates per-experiment result metrics and computes aggregate statistics.

    Thin façade over the module-level accumulator helpers: holds the dict produced
    by create_metric_accumulators() and delegates to append_result_metrics() /
    compute_aggregate_stats(), so behavior matches the original iDLG_mask.py inline
    calls exactly.
    """

    def __init__(self):
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
```

- [ ] **Step 4: Create `stable_ginv/experiment/__init__.py`**

Note: this lists `results` exports only. The `runner` import is added in Task 4 (runner.py does not exist yet).

```python
from stable_ginv.experiment.results import (
    METRIC_ACCUMULATOR_KEYS,
    EMPTY_PAIRED_STATS,
    EXP_RESULT_EXTRA_FIELDS,
    ResultAggregator,
    RestartSelector,
    create_metric_accumulators,
    append_result_metrics,
    print_result_summary,
    running_best,
    merge_restart_results,
    empty_paired_report,
    paired_report_for_both,
    paired_report_for_masked,
    compute_aggregate_stats,
    ordered_idlg_baseline_lists,
    ordered_masked_registry_lists,
    grad_param_value,
    build_common_csv_fields,
    build_exp_result_rows,
    write_masking_sweep_mse_csv,
    print_final_experiment_summary,
)

__all__ = [
    "METRIC_ACCUMULATOR_KEYS",
    "EMPTY_PAIRED_STATS",
    "EXP_RESULT_EXTRA_FIELDS",
    "ResultAggregator",
    "RestartSelector",
    "create_metric_accumulators",
    "append_result_metrics",
    "print_result_summary",
    "running_best",
    "merge_restart_results",
    "empty_paired_report",
    "paired_report_for_both",
    "paired_report_for_masked",
    "compute_aggregate_stats",
    "ordered_idlg_baseline_lists",
    "ordered_masked_registry_lists",
    "grad_param_value",
    "build_common_csv_fields",
    "build_exp_result_rows",
    "write_masking_sweep_mse_csv",
    "print_final_experiment_summary",
]
```

- [ ] **Step 5: Create the re-export shim at `functions/experiment_results.py`**

This is a brand-new file at the old path (the original content now lives in `results.py`). It re-exports the full public surface so `manual_stats.py`, `tests/test_registry_append.py`, and `iDLG_mask.py` (until Task 4) keep importing unchanged.

```python
"""Shim: experiment result aggregation / CSV moved to stable_ginv.experiment.results (Phase 6).

Re-exports the public surface so existing callers (manual_stats.py, the test suite,
iDLG_mask.py until its own move) keep working. New code should import from
stable_ginv.experiment.
"""
from stable_ginv.experiment.results import (
    METRIC_ACCUMULATOR_KEYS,
    EMPTY_PAIRED_STATS,
    EXP_RESULT_EXTRA_FIELDS,
    ResultAggregator,
    RestartSelector,
    create_metric_accumulators,
    append_result_metrics,
    print_result_summary,
    running_best,
    merge_restart_results,
    empty_paired_report,
    paired_report_for_both,
    paired_report_for_masked,
    compute_aggregate_stats,
    ordered_idlg_baseline_lists,
    ordered_masked_registry_lists,
    grad_param_value,
    build_common_csv_fields,
    build_exp_result_rows,
    write_masking_sweep_mse_csv,
    print_final_experiment_summary,
)

__all__ = [
    "METRIC_ACCUMULATOR_KEYS",
    "EMPTY_PAIRED_STATS",
    "EXP_RESULT_EXTRA_FIELDS",
    "ResultAggregator",
    "RestartSelector",
    "create_metric_accumulators",
    "append_result_metrics",
    "print_result_summary",
    "running_best",
    "merge_restart_results",
    "empty_paired_report",
    "paired_report_for_both",
    "paired_report_for_masked",
    "compute_aggregate_stats",
    "ordered_idlg_baseline_lists",
    "ordered_masked_registry_lists",
    "grad_param_value",
    "build_common_csv_fields",
    "build_exp_result_rows",
    "write_masking_sweep_mse_csv",
    "print_final_experiment_summary",
]
```

- [ ] **Step 6: Verify the façades and re-exports behave**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -c "
from stable_ginv.experiment.results import ResultAggregator, RestartSelector, create_metric_accumulators, merge_restart_results
agg = ResultAggregator()
assert agg.accumulators == create_metric_accumulators(), 'accumulator keys drift'
assert callable(agg.append) and callable(agg.aggregate_stats)
# RestartSelector.merge is a pass-through to merge_restart_results:
rd = {0: {'idx_net': 0, 'device_id': 0, 'gt_data': 1, 'gt_label': 2, 'imidx_list': []}}
assert RestartSelector.merge(rd) == merge_restart_results(rd)
# Shim re-exports resolve to the canonical objects:
import functions.experiment_results as shim
from stable_ginv.experiment import results as canon
assert shim.build_exp_result_rows is canon.build_exp_result_rows
assert shim.ResultAggregator is canon.ResultAggregator
print('experiment.results façades + shim OK')
"
```
Expected: prints `experiment.results façades + shim OK`. No error.

- [ ] **Step 7: Run the full suite (CSV golden + test_registry_append flow through the shim → identical)**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -m pytest tests/ -q
```
Expected: all tests PASS, including `tests/golden/test_experiment_results_csv_golden.py` (imports `build_*` via the shim → resolves to `stable_ginv.experiment.results`) and `tests/test_registry_append.py` (imports `ordered_masked_registry_lists`/`paired_report_for_masked` via the shim). If the CSV golden differs, STOP and debug — do not regenerate the fixture.

- [ ] **Step 8: Lint**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -m pyflakes stable_ginv/experiment functions/experiment_results.py
```
Expected: no output. (The shim re-exports names used elsewhere; `__all__` documents them as the public surface — the same pattern as `functions/io_utils.py`. If pyflakes is unavailable, use `conda run -n stable-ginv python -m py_compile stable_ginv/experiment/*.py functions/experiment_results.py`.)

- [ ] **Step 9: Commit (`refactor:`)**

```bash
git add stable_ginv/experiment/ functions/experiment_results.py
git commit -m "refactor: extract experiment_results into stable_ginv/experiment/results"
```

---

## Task 3: Point the CSV-row golden at the canonical import path

**Files:**
- Modify: `tests/golden/test_experiment_results_csv_golden.py`

Make the new dependency explicit (no longer routed through the shim), matching how Phase 4/5 repointed their tests at canonical `stable_ginv` paths.

- [ ] **Step 1: Update the imports**

Find:
```python
from functions.experiment_results import (
    build_common_csv_fields,
    build_exp_result_rows,
)
from functions.io_utils import append_csv_rows
```
Replace with:
```python
from stable_ginv.experiment.results import (
    build_common_csv_fields,
    build_exp_result_rows,
)
from stable_ginv.io import append_csv_rows
```

- [ ] **Step 2: Run the golden — bytes must be unchanged**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -m pytest tests/golden/test_experiment_results_csv_golden.py -q
```
Expected: 1 passed. The underlying functions are byte-identical via the verbatim move, so the fixture matches.

- [ ] **Step 3: Commit (`test:`)**

```bash
git add tests/golden/test_experiment_results_csv_golden.py
git commit -m "test: import experiment_results CSV golden from canonical path"
```

---

## Task 4: Extract `BatchExperimentRunner` and move `main()` into `stable_ginv/cli/batch.py`

**Files:**
- Create: `stable_ginv/experiment/runner.py`
- Modify: `stable_ginv/experiment/__init__.py`
- Create: `stable_ginv/cli/__init__.py`
- Create: `stable_ginv/cli/batch.py`
- Modify: `iDLG_mask.py`

This relocates the scheduling section of `iDLG_mask.py` **verbatim** into `BatchExperimentRunner.run()` and the rest of `main()` **verbatim** into `cli/batch.py`, with only the documented structural edits. There is no end-to-end golden for the full CLI run (it resolves to the shared `/work3` tree and needs a GPU/dataset), so correctness rests on the verbatim discipline plus the guards in Steps 6–10.

- [ ] **Step 1: Create `stable_ginv/experiment/runner.py`**

The body of `run()` is the scheduling section of `iDLG_mask.py` (the `# -------- Run experiments in parallel --------` block through the serial branch), moved verbatim with these documented edits: the `_handle_result`/`_abort_run` closures become a passed-in `handle_result` callback and a `self._abort_run` method; `active_processes` is `self.active_processes`; `run_single_experiment` is imported canonically; `merge_restart_results(...)` becomes `RestartSelector.merge(...)`; and the **final** `flush_recon_panel(...)` call at the end of *each* branch is **dropped** (hoisted to `cli/batch.py` after `run()` returns — both branches called it identically, so one post-run flush is behavior-identical).

```python
"""BatchExperimentRunner: GPU scheduling and multiprocessing for the experiment sweep.

Extracted verbatim from iDLG_mask.py (Phase 6). Owns worker scheduling, output
ordering (per-experiment, and round-robin restart interleaving when experiments are
fewer than GPUs), restart merging, and abort-on-worker-failure. Reconstruction
numerics live in stable_ginv.recon; result aggregation/CSV in
stable_ginv.experiment.results. The caller supplies a `handle_result` callback that
receives each completed experiment result in order.
"""
import dataclasses

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from stable_ginv.recon import run_single_experiment
from stable_ginv.experiment.results import RestartSelector


class ExperimentRunAborted(RuntimeError):
    """Raised after a worker failure has caused the run to be cancelled and cleaned up."""


class BatchExperimentRunner:
    """Schedules per-experiment reconstruction workers across the available GPUs."""

    def __init__(self, config, dst, dataset, num_exp, num_restarts):
        self.config = config
        self.dst = dst
        self.dataset = dataset
        self.num_exp = num_exp
        self.num_restarts = num_restarts
        self.active_processes = {}

    def _abort_run(self, result):
        idx = result.get('idx_net')
        dev = result.get('device_id')
        r_i = result.get('restart_idx')
        where = f"exp {idx}" + (f" restart {r_i}" if r_i is not None else "") + f" on GPU {dev}"
        print(f"\n[ABORT] {where} failed — cancelling entire run, no results will be saved.")
        print(result.get('traceback', result.get('error', '')))
        for proc in self.active_processes.values():
            if proc.is_alive():
                proc.terminate()
        for proc in self.active_processes.values():
            proc.join()
        raise ExperimentRunAborted(
            f"Run cancelled: {where} failed with: {result.get('error', 'unknown error')}"
        )

    def run(self, handle_result):
        config = self.config
        dst = self.dst
        dataset = self.dataset
        num_exp = self.num_exp
        NUM_RESTARTS = self.num_restarts

        # -------- Run experiments in parallel --------
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            print("[WARNING] No CUDA GPU detected — falling back to CPU with a single "
                  "worker. This is very slow and intended only for testing; a GPU is "
                  "strongly recommended for real experiments.")
            num_workers = 1
        else:
            print(f"Using {num_gpus} GPUs")
            num_workers = num_gpus

        # Set parallel worker method
        mp.set_start_method('spawn', force=True)
        mp.set_sharing_strategy('file_system')

        result_queue = mp.SimpleQueue()
        active_processes = self.active_processes

        parallel_restarts = num_exp < num_gpus and NUM_RESTARTS > 1

        if parallel_restarts:
            print(f"[INFO] Parallel restart mode: {num_exp} exp × {NUM_RESTARTS} restarts across {num_gpus} GPUs")
            # Round-robin across images so all experiments get restarts interleaved —
            # prevents 3 GPUs idling while only one image's final restart is running.
            tasks = [(exp_i, r_i) for r_i in range(NUM_RESTARTS) for exp_i in range(num_exp)]
            total_tasks = len(tasks)
            restart_buf = {exp_i: {} for exp_i in range(num_exp)}
            next_task = 0
            completed_tasks = 0

            for device_id in range(min(num_workers, total_tasks)):
                exp_i, r_i = tasks[next_task]
                task_cfg = dataclasses.replace(config, single_restart_idx=r_i)
                p = mp.Process(target=run_single_experiment,
                               args=(exp_i, device_id, dst, dataset, task_cfg, result_queue))
                p.start()
                active_processes[device_id] = p
                next_task += 1

            with tqdm(total=total_tasks, desc="Restart tasks", position=0) as pbar:
                while completed_tasks < total_tasks:
                    result = result_queue.get()
                    completed_tasks += 1
                    pbar.update(1)
                    finished_device = result['device_id']
                    active_processes[finished_device].join()

                    if result.get('error') is not None:
                        self._abort_run(result)
                    exp_i = result['idx_net']
                    r_i = result.get('restart_idx', 0)
                    restart_buf[exp_i][r_i] = result

                    if next_task < total_tasks:
                        exp_i, r_i = tasks[next_task]
                        task_cfg = dataclasses.replace(config, single_restart_idx=r_i)
                        p = mp.Process(target=run_single_experiment,
                                       args=(exp_i, finished_device, dst, dataset, task_cfg, result_queue))
                        p.start()
                        active_processes[finished_device] = p
                        next_task += 1

            for p in active_processes.values():
                p.join()

            for exp_i in range(num_exp):
                if restart_buf[exp_i]:
                    handle_result(RestartSelector.merge(restart_buf[exp_i]))

        else:
            next_exp = 0
            completed = 0

            # Start one experiment per worker initially (one per GPU, or one on CPU)
            for device_id in range(min(num_workers, num_exp)):
                p = mp.Process(
                    target=run_single_experiment,
                    args=(next_exp, device_id, dst, dataset, config, result_queue)
                )
                p.start()
                active_processes[device_id] = p
                next_exp += 1

            # Keep launching a new experiment whenever one finishes
            with tqdm(total=num_exp, desc="Experiments", position=0) as pbar:
                while completed < num_exp:
                    result = result_queue.get()
                    completed += 1
                    pbar.update(1)
                    pbar.set_postfix(last_exp=result["idx_net"], gpu=result["device_id"])

                    finished_device = result['device_id']

                    if result.get('error') is not None:
                        self._abort_run(result)

                    handle_result(result)

                    # Clean up the finished process on that GPU
                    active_processes[finished_device].join()

                    # Start the next experiment immediately on the freed GPU
                    if next_exp < num_exp:
                        p = mp.Process(
                            target=run_single_experiment,
                            args=(next_exp, finished_device, dst, dataset, config, result_queue)
                        )
                        p.start()
                        active_processes[finished_device] = p
                        tqdm.write(f"Launching experiment {next_exp} on GPU {finished_device}")
                        next_exp += 1

            # Final cleanup
            for p in active_processes.values():
                p.join()
```

- [ ] **Step 2: Add the runner exports to `stable_ginv/experiment/__init__.py`**

Find:
```python
from stable_ginv.experiment.results import (
```
Insert **above** that line:
```python
from stable_ginv.experiment.runner import BatchExperimentRunner, ExperimentRunAborted
```
Then in the `__all__` list, add the two names at the top (after the opening `[`):
```python
    "BatchExperimentRunner",
    "ExperimentRunAborted",
```

> Import-order note: `runner` imports `from stable_ginv.experiment.results import RestartSelector`. Importing `runner` first triggers `results` to load fully, then binds `RestartSelector` — acyclic, no circular-import risk.

- [ ] **Step 3: Create `stable_ginv/cli/__init__.py`**

Keep it side-effect free (the wrapper imports `stable_ginv.cli.batch` directly):

```python
"""CLI entry points for stable_ginv (Phase 6+)."""
```

- [ ] **Step 4: Create `stable_ginv/cli/batch.py`**

Start from a copy of the current `iDLG_mask.py`, then apply the edits below:

```bash
cp /home/mathias/GitHub/stable-ginv/iDLG_mask.py /home/mathias/GitHub/stable-ginv/stable_ginv/cli/batch.py
```

Edit 4a — replace the header + import block. Find (lines 1–40, from `# iDLG_mask.py` through `from stable_ginv.config import ExperimentConfig`):
```python
# iDLG_mask.py
import os
import sys
import torch
from torchvision import transforms
from datetime import datetime

import torch.multiprocessing as mp
from functions.experiment_results import (
    append_result_metrics,
    build_common_csv_fields,
    build_exp_result_rows,
    compute_aggregate_stats,
    create_metric_accumulators,
    empty_paired_report,
    grad_param_value,
    merge_restart_results,
    ordered_idlg_baseline_lists,
    ordered_masked_registry_lists,
    paired_report_for_both,
    paired_report_for_masked,
    print_final_experiment_summary,
    print_result_summary,
    write_masking_sweep_mse_csv,
)
from functions.idlg_cli import parse_idlg_args
from functions.io_utils import (baseline_key_from_args,
    load_baseline_registry, save_baseline_registry, update_idlg_baseline,
    write_baseline_summary_csv, parse_prefixes_with_fracs, masked_key_from_args,
    load_masked_registry, save_masked_registry, update_masked_registry,
    append_csv_rows, find_registry_entry, resolve_storage_paths, safe_makedirs,
    seed_registry_entry_from_fallback)
from functions.Dataset import load_dataset
from run_single_exp import run_single_experiment
from tqdm import tqdm

from functions.io_utils import setstdout

import dataclasses
from stable_ginv.config import ExperimentConfig
```
Replace with:
```python
"""iDLG masked gradient-inversion experiment entry point (Phase 6).

Orchestration moved here from iDLG_mask.py: argument parsing, data loading,
ExperimentConfig construction, scheduling via BatchExperimentRunner, result
aggregation, registry/paired statistics, CSV output, and the final summary.
iDLG_mask.py is now a thin wrapper around main().
"""
import os
import sys
from datetime import datetime

import torch
from torchvision import transforms
from tqdm import tqdm

from functions.idlg_cli import parse_idlg_args
from functions.Dataset import load_dataset
from stable_ginv.config import ExperimentConfig
from stable_ginv.io import (
    append_csv_rows,
    parse_prefixes_with_fracs,
    resolve_storage_paths,
    safe_makedirs,
    setstdout,
)
from stable_ginv.registry import (
    baseline_key_from_args,
    find_registry_entry,
    load_baseline_registry,
    load_masked_registry,
    masked_key_from_args,
    save_baseline_registry,
    save_masked_registry,
    seed_registry_entry_from_fallback,
    update_idlg_baseline,
    update_masked_registry,
    write_baseline_summary_csv,
)
from stable_ginv.experiment.results import (
    ResultAggregator,
    build_common_csv_fields,
    build_exp_result_rows,
    empty_paired_report,
    grad_param_value,
    ordered_idlg_baseline_lists,
    ordered_masked_registry_lists,
    paired_report_for_both,
    paired_report_for_masked,
    print_final_experiment_summary,
    print_result_summary,
    write_masking_sweep_mse_csv,
)
from stable_ginv.experiment.runner import BatchExperimentRunner
```

Edit 4b — delete the `ExperimentRunAborted` class (it now lives in `runner.py`). Find and delete:
```python
class ExperimentRunAborted(RuntimeError):
    """Raised after a worker failure has caused the run to be cancelled and cleaned up."""


```

> `_load_visualization_helpers()` is left exactly as-is (it loads `helper.visualization` lazily; that move is Phase 7).

Edit 4c — swap the accumulator init for the façade. Find:
```python
    metric_accumulators = create_metric_accumulators()
```
Replace with:
```python
    aggregator = ResultAggregator()
```

Edit 4d — replace the entire scheduling section with the runner call + one hoisted final flush. Find the whole block from:
```python
    # -------- Run experiments in parallel --------
    num_gpus = torch.cuda.device_count()
```
…through the end of the `else` branch's final cleanup + flush:
```python
        # Final cleanup
        for p in active_processes.values():
            p.join()
        # save any remaining
        panel_block_idx = flush_recon_panel(
            params, panel_buffers, panel_png_paths, save_path, panel_block_idx,
            dataset, mask_desc, timestamp_str, METHODS,
        )
```
Replace that entire block with:
```python
    # -------- Run experiments in parallel --------
    all_results_by_idx = {}

    # ---- _handle_result callback: accumulates lists, builds panel, prints ----
    def _handle_result(result):
        nonlocal panel_block_idx
        idx = result["idx_net"]
        all_results_by_idx[idx] = result

        aggregator.append(result)
        append_result_to_panel_buffers(result, panel_buffers, tp, tqdm.write)
        if len(panel_buffers["gt"]) == panel_block_size:
            panel_block_idx = flush_recon_panel(
                params, panel_buffers, panel_png_paths, save_path, panel_block_idx,
                dataset, mask_desc, timestamp_str, METHODS,
            )

        print_result_summary(result)

    runner = BatchExperimentRunner(config, dst, dataset, num_exp, NUM_RESTARTS)
    runner.run(_handle_result)

    # Flush any remaining buffered panels (both scheduling branches ended with this).
    panel_block_idx = flush_recon_panel(
        params, panel_buffers, panel_png_paths, save_path, panel_block_idx,
        dataset, mask_desc, timestamp_str, METHODS,
    )
```

Edit 4e — point the two post-run accumulator reads at the façade. Find:
```python
    psnr_per_restart_idlg_all = metric_accumulators["psnr_per_restart_idlg"]
    psnr_per_restart_masked_all = metric_accumulators["psnr_per_restart_masked"]
    mse_per_restart_idlg_all = metric_accumulators["mse_per_restart_idlg"]
    mse_per_restart_masked_all = metric_accumulators["mse_per_restart_masked"]
```
Replace with:
```python
    psnr_per_restart_idlg_all = aggregator.accumulators["psnr_per_restart_idlg"]
    psnr_per_restart_masked_all = aggregator.accumulators["psnr_per_restart_masked"]
    mse_per_restart_idlg_all = aggregator.accumulators["mse_per_restart_idlg"]
    mse_per_restart_masked_all = aggregator.accumulators["mse_per_restart_masked"]
```

Edit 4f — the masked-registry guard read. Find:
```python
    if METHODS in ["masked", "both"] and metric_accumulators["best_psnr_masked"]:
```
Replace with:
```python
    if METHODS in ["masked", "both"] and aggregator.accumulators["best_psnr_masked"]:
```

Edit 4g — the aggregate-stats call. Find:
```python
    stats = compute_aggregate_stats(metric_accumulators)
```
Replace with:
```python
    stats = aggregator.aggregate_stats()
```

Edit 4h — delete the trailing `if __name__ == '__main__':` guard at the bottom of `batch.py` (the entry point stays in the root wrapper). Find and delete:
```python


if __name__ == '__main__':
    main()
```

> After these edits, `batch.py` no longer references `mp`, `dataclasses`, `run_single_experiment`, `create_metric_accumulators`, `append_result_metrics`, `compute_aggregate_stats`, `merge_restart_results`, or `ExperimentRunAborted` — all are encapsulated in the runner or the façade, which is why they were dropped from the imports in Edit 4a.

- [ ] **Step 5: Replace `iDLG_mask.py` with the thin wrapper**

```python
"""Thin entry point: orchestration lives in stable_ginv.cli.batch (Phase 6).

Kept at the repo root so existing HPC commands (`python iDLG_mask.py ...`) and job
scripts keep working unchanged. New code should import from stable_ginv.cli.batch.
"""
from stable_ginv.cli.batch import main

if __name__ == '__main__':
    main()
```

- [ ] **Step 6: Verify imports, wrapper identity, and spawn target resolvability**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -c "
from stable_ginv.cli.batch import main
import iDLG_mask
assert iDLG_mask.main is main, 'wrapper does not re-export the canonical main'
from stable_ginv.experiment import BatchExperimentRunner, ExperimentRunAborted
from stable_ginv.experiment.runner import BatchExperimentRunner as R2
assert BatchExperimentRunner is R2
# Spawn target unchanged from Phase 5 (pickled by module+qualname for the child):
from stable_ginv.recon import run_single_experiment
print('spawn target:', run_single_experiment.__module__, run_single_experiment.__qualname__)
print('cli.batch + experiment.runner wiring OK')
"
```
Expected: prints `spawn target: stable_ginv.recon.runner run_single_experiment` then `cli.batch + experiment.runner wiring OK`. No error.

- [ ] **Step 7: Confirm the CLI help is byte-identical**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python iDLG_mask.py --help
```
Expected: the full argument help prints and the process exits 0 (the parser in `functions/idlg_cli.py` is untouched).

- [ ] **Step 8: Run the full suite**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -m pytest tests/ -q
```
Expected: all tests PASS (no test imports `iDLG_mask`'s body; the experiment goldens and registry tests still pass).

- [ ] **Step 9: Review the diff — `iDLG_mask.py` should have shrunk to the wrapper**

```bash
git add -A
git diff --cached --stat
```
Expected: `iDLG_mask.py` shows a large deletion (down to ~9 lines); `stable_ginv/cli/batch.py`, `stable_ginv/cli/__init__.py`, `stable_ginv/experiment/runner.py` are added; `stable_ginv/experiment/__init__.py` is modified. Eyeball `git diff --cached stable_ginv/cli/batch.py` against the original `main()` body to confirm only the documented edits were applied.

- [ ] **Step 10: Lint and whitespace check**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -m pyflakes stable_ginv/cli stable_ginv/experiment iDLG_mask.py
git diff --cached --check
```
Expected: no pyflakes output (all imports in `batch.py`/`runner.py` are used; see the Edit-4a note), no whitespace errors. (If pyflakes is unavailable, use `conda run -n stable-ginv python -m py_compile stable_ginv/cli/*.py stable_ginv/experiment/*.py iDLG_mask.py`.)

- [ ] **Step 11: Commit (`refactor:`)**

```bash
git add stable_ginv/cli/ stable_ginv/experiment/ iDLG_mask.py
git commit -m "refactor: move iDLG_mask orchestration into stable_ginv/cli/batch + experiment/runner"
```

---

## Task 5: Update the handover and finalize

**Files:**
- Modify: `docs/handover/HANDOVER_RESTRUCTURE.md`

- [ ] **Step 1: Mark Phase 6 complete in the Status section**

Find:
```markdown
- [ ] Phase 6 — `stable_ginv/experiment/` (+ CSV-row goldens first).
```
Replace with:
```markdown
- [x] Phase 6 — `stable_ginv/experiment/` (BatchExperimentRunner, ResultAggregator,
      RestartSelector, results.py) + `stable_ginv/cli/batch.py`. `iDLG_mask.py` is a
      thin wrapper; `functions/experiment_results.py` is a shim. CSV-row golden added.
```

- [ ] **Step 2: Replace the `## Next phase` section**

Find the entire `## Next phase` block and replace it with:
```markdown
## Next phase

Write the Phase 7 plan from the spec, then implement. Phase 7 extracts
`stable_ginv/viz/` from `helper/visualization.py` (recon panels/GIFs, restart curves)
and the standalone `helper/plot_*.py` CLIs, replacing them with thin wrappers. Lock
the relevant plot outputs with goldens just-in-time before moving them (the existing
`tests/test_visualization.py` / `tests/test_restart_curve.py` are the starting point).
`stable_ginv/cli/batch.py` loads visualization lazily via `_load_visualization_helpers`,
so repoint that import at `stable_ginv.viz` once the package exists. Keep this file's
Status section current at every phase boundary.

Deferred (not yet done): `functions/idlg_cli.py` → `stable_ginv/cli/args.py` and
`functions/Dataset.py`/`functions/consts.py` → `stable_ginv/data/` remain imported from
their current paths by `stable_ginv/cli/batch.py` and `stable_ginv/recon/runner.py`.
```

- [ ] **Step 3: Final full verification**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -m pytest tests/ -q
conda run -n stable-ginv python iDLG_mask.py --help
git diff --check
```
Expected: all tests PASS, help prints, no whitespace errors.

- [ ] **Step 4: Commit (`docs:`)**

```bash
git add docs/handover/HANDOVER_RESTRUCTURE.md
git commit -m "docs: mark Phase 6 complete in HANDOVER_RESTRUCTURE"
```

---

## Self-review checklist

**Spec coverage (Section 1 mapping `iDLG_mask.py` → `cli/batch.py` + `experiment/runner.py`; `functions/experiment_results.py` → `experiment/results.py` + `io/csv.py`):**
- [x] `BatchExperimentRunner` owns GPU scheduling, multiprocessing, output ordering, abort-on-worker-failure (Task 4, verbatim from the scheduling section).
- [x] `ResultAggregator` (aggregation) + `RestartSelector` (restart merging) façades (Task 2, wired into `cli/batch.py` and `runner.py` in Task 4).
- [x] `functions/experiment_results.py` → `stable_ginv/experiment/results.py` (Task 2). The "+ `stable_ginv/io/csv.py`" half of the spec mapping was already satisfied in Phase 4 (`append_csv_rows`/`append_csv_row` live in `io/csv.py`); `results.py` imports the writer from there.
- [x] `iDLG_mask.py` → thin `from stable_ginv.cli.batch import main` wrapper, kept importable for HPC commands (Task 4 Step 5; wrapper identity verified Step 6).
- [x] Multiprocessing spawn target stays `stable_ginv.recon.runner.run_single_experiment` (Task 4 Step 6).
- [x] Per-run `experiment_results` CSV-row golden added **first**, before the refactor (Task 1), per spec Section 2.3.
- [x] Categorized commits: `test:` (golden lock) / `refactor:` (results extract) / `test:` (golden repoint) / `refactor:` (orchestration move) / `docs:`.
- [x] `HANDOVER_RESTRUCTURE.md` Status + Next phase updated (Task 5).

**Placeholder scan:** no TBD/TODO/"handle edge cases"/"similar to". `runner.py` and `cli/batch.py` are full verbatim relocations with every edit spelled out as find/replace; `results.py` façades and both shims are complete.

**Type/name consistency:**
- [x] `ResultAggregator()` / `.accumulators` / `.append(result)` / `.aggregate_stats()` are referenced identically in Task 4 (Edits 4c–4g) as defined in Task 2 Step 3.
- [x] `RestartSelector.merge(rdict)` defined Task 2, called in `runner.run` (Task 4 Step 1) and asserted equal to `merge_restart_results` (Task 2 Step 6).
- [x] `BatchExperimentRunner(config, dst, dataset, num_exp, num_restarts)` + `.run(handle_result)` defined Task 4 Step 1, called with exactly those positional args in `cli/batch.py` (Edit 4d).
- [x] `_handle_result(result)` is the single callback passed to `runner.run`; it closes over `panel_block_idx`/`panel_buffers`/`aggregator`/`all_results_by_idx`, all assigned in `main` before the call.
- [x] Shim re-export list, `experiment/__init__.py` `__all__`, and the names imported by `cli/batch.py` all draw from the same 19-name public surface enumerated in Task 2.

**Potential issues to watch:**
- **No end-to-end golden for the orchestration:** a full CLI run writes to the shared `/work3` tree and needs a GPU/dataset, so it is not run. Correctness rests on (a) the verbatim move, (b) the CSV-row golden guarding the row/column output, (c) the registry/baseline goldens guarding registry writes, (d) `--help`, and (e) the import/wiring checks in Task 4 Step 6. The diff-review step (Task 4 Step 9) is the backstop — confirm only documented edits were applied.
- **Hoisted panel flush:** both scheduling branches ended with an identical `flush_recon_panel(...)`. The runner drops both; `cli/batch.py` calls it once after `runner.run` returns. Intermediate block flushes still happen inside `_handle_result`. Net behavior is identical.
- **`metric_accumulators` references:** the original has exactly 7 (init at 180, inside `_handle_result`, four restart-curve reads, one masked-registry guard, one `compute_aggregate_stats` call). All are covered by Edits 4c/4d/4e/4f/4g — grep `cli/batch.py` for `metric_accumulators` after editing and expect **zero** hits.
- **Circular imports:** `experiment.results` imports only `stable_ginv.io`/`stable_ginv.stats` (+ stdlib/numpy); `experiment.runner` imports `stable_ginv.recon` + `experiment.results`; `cli.batch` imports `experiment.results` + `experiment.runner` + `config`/`io`/`registry`. None import back into `cli` or `experiment.__init__`, so the graph is acyclic.
- **Kept imports:** `parse_idlg_args` (from `functions.idlg_cli`), `load_dataset` (from `functions.Dataset`), and the lazy `helper.visualization` load stay on their current paths — their package moves are later phases (noted in the handover Next-phase block). Do not move them here.
- **Do not touch:** reconstruction math, masking, optimizer branches, registry key inputs/JSON, CSV columns/order/rounding, or any `print(...)` string literal — all are inside verbatim-moved bodies.
```