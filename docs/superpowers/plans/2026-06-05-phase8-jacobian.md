# Phase 8 — `stable_ginv/jacobian/` Sweep + Rank-vs-Reconstruction Plot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract `functions/jacobian_rank_sweep.py` (Jacobian rank sweep + its CLI) into a new `stable_ginv/jacobian/` package (`sweep.py` + `cli.py`), and move `functions/rank_reconstruction_plot.py` into `stable_ginv/viz/plot_rank_reconstruction.py`, replacing both `functions/` files with thin runnable wrappers — with **zero** change to CLI flags/defaults, `--help` text, CSV columns/order, printed strings, plot filenames/contents, or sweep numerics.

**Architecture:** Mirrors the Phase 2/4/5/6/7 shim-and-relocate discipline. `functions/jacobian_rank_sweep.py` (625 lines) splits by responsibility: `stable_ginv/jacobian/sweep.py` holds the rank-sweep compute core (device/seed/dtype helpers, `_worker_core`, the `mp.spawn` target `_mp_worker`, the serial/parallel runners, and the result-summarization helpers), and `stable_ginv/jacobian/cli.py` holds the argparse `main()` (validation, CSV writer, Seaborn plot) plus the `_parse_explicit_sample_indices` arg helper. The multiprocessing worker (`_mp_worker`) lives in `sweep.py` so `mp.spawn` re-imports it from a stable package path. `functions/rank_reconstruction_plot.py` moves **whole** into `stable_ginv/viz/plot_rank_reconstruction.py`. Both `functions/` files become thin wrappers that re-add the repo root to `sys.path` and call `main`. Code bodies are relocated **verbatim**; the only edits are the documented import repointings, dropping the now-misdirected `sys.path` bootstrap, and (in a **separate** commit) fixing the pre-existing f-string-without-placeholder lint.

**Tech Stack:** Python 3, PyTorch + `torch.multiprocessing`, torchvision, NumPy, pandas, Matplotlib (Agg backend), Seaborn, tqdm, SciPy (via metrics), pytest. Conda env: `stable-ginv`. All verification via `conda run -n stable-ginv`. Always `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"` before Python (DTU HPC C++ runtime requirement).

---

## Project commit rule (read before any `git commit`)

This project's standing rule (and the agent's stored memory): **never commit automatically — create a commit only when the user explicitly asks**, and **never add `Co-Authored-By:` or any co-author trailer**. Every "Commit" step below documents the exact categorized message to use *when the user authorizes it*. Do not run `git commit` without that authorization. Categorized commits (no `feat:`): `refactor:` (code moves), `test:` (goldens/test repoints), `style:` (lint fix), `docs:` (handover). Separate commit per category.

## Invariants (enforce every task)

- `conda run -n stable-ginv python -m pytest tests/ -q` stays green throughout, including all existing goldens and the **two new** goldens added in Tasks 1–2.
- `conda run -n stable-ginv python iDLG_mask.py --help` works throughout (untouched by this phase).
- These keep working with byte-identical `--help` output before and after the move:
  - `conda run -n stable-ginv python functions/jacobian_rank_sweep.py --help`
  - `conda run -n stable-ginv python -m functions.jacobian_rank_sweep --help`
  - `conda run -n stable-ginv python -m functions.rank_reconstruction_plot --help`
- No change to: CLI flags/defaults/choices, argparse validation messages, the sweep CSV header/column order/row values, any `print(...)`/`tqdm.write(...)` string literal, the Seaborn plot construction, output filenames, or sweep/rank numerics.
- Moved bodies are relocated **verbatim**. The only permitted edits are the documented ones in each task: import repointing, dropping the misdirected `sys.path.insert` bootstrap, splitting the file across `sweep.py`/`cli.py` at the documented boundary, and (Task 7 only, separate commit) the f-string lint fix. Do not reorder, reformat, rename, or "improve" any relocated function body.
- **Never regenerate a golden to make a refactor pass.** If any golden drifts after the move, STOP and debug — the move was not verbatim. Do not edit any fixture except via `GOLDEN_REGEN=1` at first creation (Tasks 1–2).
- The cache-dir env bootstrap (`MPLCONFIGDIR`/`XDG_CACHE_HOME`) must still run in the spawned worker process. It is therefore placed at the top of `sweep.py` (the module `mp.spawn` re-imports), and runs before any Matplotlib import in `cli.py` (which imports `sweep` first).

## Module boundaries (current `functions/jacobian_rank_sweep.py` → target)

| Target | Content (verbatim from `functions/jacobian_rank_sweep.py`) |
|--------|-----------------------------------------------------------|
| `stable_ginv/jacobian/sweep.py` | Cache-dir env bootstrap (lines 14–20); module constant `_TORCH_DTYPES`; helpers `_get_last_fc_param_indices`, `_split_list`, `_torch_dtype`, `_seed_all`, `_resolve_device`, `_new_rank_results`, `_summarize_rank_results`, `_print_rank_summary`, `_print_mask_debug`; `_worker_core`; `_mp_worker`; `_run_serial`; `_run_parallel`; `_run_dtype` |
| `stable_ginv/jacobian/cli.py` | `_parse_explicit_sample_indices`; `main` |
| `stable_ginv/jacobian/__init__.py` | Empty package marker (subpackage; nothing re-exported) |
| `functions/jacobian_rank_sweep.py` | Thin runnable wrapper → `from stable_ginv.jacobian.cli import main` |

## Module boundaries (`functions/rank_reconstruction_plot.py` → target)

| Target | Content |
|--------|---------|
| `stable_ginv/viz/plot_rank_reconstruction.py` | Verbatim `functions/rank_reconstruction_plot.py` (module docstring kept; imports repointed; f-string lint fixed in Task 7) |
| `functions/rank_reconstruction_plot.py` | Thin runnable wrapper → `from stable_ginv.viz.plot_rank_reconstruction import main` |

### Exact import repointings

**`sweep.py`** (was `functions/jacobian_rank_sweep.py` lines 1–38):
- DROP line 1–2 `sys.path.insert(...)` bootstrap (editable install + wrapper handle path). Keep `import sys` (used by `tqdm.write(..., file=sys.stderr)`) and `import os` (used by cache bootstrap).
- KEEP the cache-dir env bootstrap block (orig lines 14–20).
- KEEP `import functions.consts as consts` and `from functions.Dataset import load_dataset` (data/ deferred to a later phase, per spec).
- KEEP `from helper.Network import get_model, weights_init` (models/ deferred, per Phase 7 precedent).
- DO NOT import matplotlib/seaborn/csv/argparse/datetime/pandas in `sweep.py` (those belong to `cli.py`). The original module-level `import matplotlib … import seaborn as sns; sns.set_theme(...)` lines and `import csv, argparse` move to `cli.py`.
- REPOINT:
  - `from functions.io_utils import (parse_prefixes_with_fracs, resolve_storage_paths, safe_makedirs, safe_savefig)` → in `sweep.py` keep only what the worker uses: `from stable_ginv.io import resolve_storage_paths` (used by `_worker_core`). `parse_prefixes_with_fracs`, `safe_makedirs`, `safe_savefig` are used only by `main` → import them in `cli.py`.
  - `from functions.masking import build_gradient_mask` → `from stable_ginv.masking import build_gradient_mask`
  - `from helper.metrics import compute_jacobian_rank_sweep` → `from stable_ginv.metrics import compute_jacobian_rank_sweep`

**`cli.py`** (was `functions/jacobian_rank_sweep.py` `main` + `_parse_explicit_sample_indices`):
- `from stable_ginv.jacobian.sweep import _run_dtype, _seed_all` **placed before the Matplotlib import** (importing `sweep` runs its cache-dir bootstrap, so `MPLCONFIGDIR`/`XDG_CACHE_HOME` are set before `import matplotlib`).
- `import os`, `import csv`, `import argparse`, `from datetime import datetime`, `import numpy as np`, `import torch.multiprocessing as mp`.
- `import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; import seaborn as sns` then `sns.set_theme(style="whitegrid")` at module level (verbatim from orig lines 22–40).
- Keep `import pandas as pd` **inline inside `main`** exactly as the original (orig line 594).
- `from stable_ginv.io import parse_prefixes_with_fracs, resolve_storage_paths, safe_makedirs, safe_savefig`
- `from functions.Dataset import load_dataset`

**`stable_ginv/viz/plot_rank_reconstruction.py`** (was `functions/rank_reconstruction_plot.py` lines 30–35):
- KEEP `import functions.consts as consts` and `from functions.Dataset import load_dataset` (data/ deferred).
- KEEP `from helper.Network import get_model, weights_init` (models/ deferred).
- REPOINT:
  - `from functions.io_utils import resolve_storage_paths` → `from stable_ginv.io import resolve_storage_paths`
  - `from functions.masking import _get_last_fc_param_indices, flatten_observed_gradients` → `from stable_ginv.masking import _get_last_fc_param_indices, flatten_observed_gradients`
  - `from helper.metrics import compute_grad_match_loss, total_variation, compute_jacobian_rank, _layer_spread_non_fc` → `from stable_ginv.metrics import compute_grad_match_loss, total_variation, compute_jacobian_rank, _layer_spread_non_fc`

## File map

| Action | Path | Purpose |
|--------|------|---------|
| Create | `tests/golden/test_jacobian_summary_golden.py` | Locks `_summarize_rank_results` (mean/std assembly that feeds the CSV + plot) before the sweep move |
| Create (auto) | `tests/golden/fixtures/jacobian_summary_golden.json` | Golden fixture (auto-generated on first run) |
| Create | `tests/golden/test_rank_reconstruction_masks_golden.py` | Locks `_build_global_topk_masks` + `_build_keepfc_masks` (topk_abs) before the rank-recon move |
| Create (auto) | `tests/golden/fixtures/rank_reconstruction_masks_golden.json` | Golden fixture (auto-generated on first run) |
| Create | `stable_ginv/jacobian/__init__.py` | Package marker |
| Create | `stable_ginv/jacobian/sweep.py` | Sweep compute core + runners (verbatim) |
| Create | `stable_ginv/jacobian/cli.py` | argparse `main()` + CSV + plot (verbatim) |
| Delete | `functions/jacobian_rank_sweep.py` (original) | Replaced by wrapper |
| Create | `functions/jacobian_rank_sweep.py` | Thin runnable wrapper |
| Move | `functions/rank_reconstruction_plot.py` → `stable_ginv/viz/plot_rank_reconstruction.py` | Verbatim relocation (imports repointed) |
| Create | `functions/rank_reconstruction_plot.py` | Thin runnable wrapper |
| Modify | `tests/test_jacobian_sweep_metadata.py` | Repoint import → `stable_ginv.jacobian.cli` |
| Modify | `tests/golden/test_jacobian_summary_golden.py` | Repoint import → `stable_ginv.jacobian.sweep` |
| Modify | `tests/golden/test_rank_reconstruction_masks_golden.py` | Repoint import → `stable_ginv.viz.plot_rank_reconstruction` |
| Modify | `stable_ginv/viz/plot_rank_reconstruction.py` | Task 7: f-string lint fix (separate commit) |
| Modify | `docs/handover/HANDOVER_RESTRUCTURE.md` | Mark Phase 8 complete; update goldens list + Next phase |

**Do not edit `pyproject.toml`.** The editable install puts the repo root on `sys.path`, so `stable_ginv.jacobian` imports as an on-disk subpackage exactly like `stable_ginv.experiment`/`.recon`/`.viz` already do. **Do not touch `stable_ginv/viz/__init__.py`** — the `plot_*` CLIs are not re-exported there (Phase 7 precedent).

---

## Task 1: Golden — lock `_summarize_rank_results`

**Files:**
- Create: `tests/golden/test_jacobian_summary_golden.py`
- Create (auto): `tests/golden/fixtures/jacobian_summary_golden.json`

`_summarize_rank_results(row_counts, rank_results, rank_results_qr)` is the pure function that turns per-sample rank lists into the `mean_ranks`/`std_ranks`/`mean_ranks_qr`/`std_ranks_qr` series that `main()` writes into the CSV and feeds to the plot. Locking it guards the verbatim relocation of the sweep's data-shaping. Inputs are synthetic (no dataset/GPU); the `_produce` callable returns a JSON-friendly dict of plain lists/floats (avoids int-keyed-dict JSON ambiguity).

- [ ] **Step 1: Write the golden test (imports the OLD path)**

```python
"""Golden test: the Jacobian-sweep mean/std summarization is numerically stable
across the Phase 8 move.

`_summarize_rank_results` converts per-sample rank lists into the mean/std series
that the sweep CLI writes to its CSV and plots. The synthetic inputs make the
output reproducible without a dataset or GPU. Imported from the OLD path here;
repointed to stable_ginv.jacobian.sweep in Task 5.
"""
from functions.jacobian_rank_sweep import _summarize_rank_results
from tests.golden.helpers import load_or_regen

# Three "samples" per row count; a separate qr pool so both branches are exercised.
_RANK_RESULTS = {
    3072: [3000, 3050, 3072],
    4000: [3500, 3600, 3700],
    5000: [4100, 4250, 4400],
}
_RANK_RESULTS_QR = {
    3072: [3010, 3040, 3070],
    4000: [3550, 3620, 3680],
    5000: [4150, 4260, 4380],
}


def _produce():
    run = _summarize_rank_results(
        list(_RANK_RESULTS.keys()), _RANK_RESULTS, _RANK_RESULTS_QR
    )
    return {
        "xs": run["xs"],
        "mean_ranks": run["mean_ranks"],
        "std_ranks": run["std_ranks"],
        "mean_ranks_qr": run["mean_ranks_qr"],
        "std_ranks_qr": run["std_ranks_qr"],
    }


def test_jacobian_summary_stable():
    golden = load_or_regen("jacobian_summary_golden.json", _produce)
    assert _produce() == golden
```

- [ ] **Step 2: Run to verify it fails (no fixture yet)**

Run: `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" && conda run -n stable-ginv python -m pytest tests/golden/test_jacobian_summary_golden.py -q`
Expected: On a fresh checkout the fixture is auto-created by `load_or_regen` and the test PASSES on first run (the helper writes then returns the value). To prove the assertion path, instead run with regen explicitly in Step 3.

- [ ] **Step 3: Generate the fixture and confirm it passes**

Run: `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" && GOLDEN_REGEN=1 conda run -n stable-ginv python -m pytest tests/golden/test_jacobian_summary_golden.py -q && conda run -n stable-ginv python -m pytest tests/golden/test_jacobian_summary_golden.py -q`
Expected: Both runs PASS. `tests/golden/fixtures/jacobian_summary_golden.json` now exists and is git-tracked.

- [ ] **Step 4: Confirm full suite still green**

Run: `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" && conda run -n stable-ginv python -m pytest tests/ -q`
Expected: PASS (all prior tests + the new golden).

- [ ] **Step 5: Commit (only when the user authorizes — no co-author trailer)**

```bash
git add tests/golden/test_jacobian_summary_golden.py tests/golden/fixtures/jacobian_summary_golden.json
git commit -m "test: lock jacobian-sweep summarization before Phase 8 move"
```

---

## Task 2: Golden — lock the rank-reconstruction mask builders

**Files:**
- Create: `tests/golden/test_rank_reconstruction_masks_golden.py`
- Create (auto): `tests/golden/fixtures/rank_reconstruction_masks_golden.json`

`_build_global_topk_masks(grads, budget)` (global top-k by |grad|) and `_build_keepfc_masks(grads, fc_ids, budget, select_mode="topk_abs")` (non-FC top-k by |grad|) are the deterministic mask-construction functions unique to the rank-vs-reconstruction CLI. Locking their boolean output guards the verbatim relocation. Synthetic CPU grads (fixed `torch.tensor` values) make the selection deterministic; the `layer_spread` path is intentionally not locked here because it delegates to `_layer_spread_non_fc` (already locked in the Phase 2 metrics goldens) and needs a real net.

- [ ] **Step 1: Write the golden test (imports the OLD path)**

```python
"""Golden test: the rank-vs-reconstruction mask builders are byte-stable across
the Phase 8 move.

`_build_global_topk_masks` and `_build_keepfc_masks` (topk_abs path) select the
top-|grad| entries that define each reconstruction budget. Synthetic CPU grads
make the selected indices deterministic. Imported from the OLD path here;
repointed to stable_ginv.viz.plot_rank_reconstruction in Task 6.
"""
import torch

from functions.rank_reconstruction_plot import (
    _build_global_topk_masks,
    _build_keepfc_masks,
)
from tests.golden.helpers import load_or_regen


def _fixed_grads():
    # Four "parameter" tensors with distinct, deterministic magnitudes.
    return [
        torch.tensor([0.5, -2.0, 1.0, 0.1]),
        torch.tensor([[3.0, -0.2], [0.05, 4.0]]),
        torch.tensor([-1.5, 0.3, 2.5]),
        torch.tensor([0.9, -0.9]),  # treated as the FC tensor below
    ]


def _masks_to_indices(entry_masks):
    """Serialize a list of bool masks to {param_index: sorted true flat-indices}."""
    out = {}
    for i, m in enumerate(entry_masks):
        if m is None:
            continue
        idx = torch.nonzero(m.reshape(-1), as_tuple=False).reshape(-1).tolist()
        out[str(i)] = sorted(int(j) for j in idx)
    return out


def _produce():
    grads = _fixed_grads()
    fc_ids = {3}  # last tensor is the FC layer
    return {
        "global_topk_b6": _masks_to_indices(_build_global_topk_masks(grads, 6)),
        "global_topk_b3": _masks_to_indices(_build_global_topk_masks(grads, 3)),
        "keepfc_topk_b4": _masks_to_indices(
            _build_keepfc_masks(grads, fc_ids, 4, select_mode="topk_abs")
        ),
    }


def test_rank_reconstruction_masks_stable():
    golden = load_or_regen("rank_reconstruction_masks_golden.json", _produce)
    assert _produce() == golden
```

- [ ] **Step 2: Generate the fixture and confirm it passes**

Run: `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" && GOLDEN_REGEN=1 conda run -n stable-ginv python -m pytest tests/golden/test_rank_reconstruction_masks_golden.py -q && conda run -n stable-ginv python -m pytest tests/golden/test_rank_reconstruction_masks_golden.py -q`
Expected: Both runs PASS. `tests/golden/fixtures/rank_reconstruction_masks_golden.json` exists and is git-tracked.

- [ ] **Step 3: Confirm full suite still green**

Run: `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" && conda run -n stable-ginv python -m pytest tests/ -q`
Expected: PASS.

- [ ] **Step 4: Commit (only when the user authorizes — no co-author trailer)**

```bash
git add tests/golden/test_rank_reconstruction_masks_golden.py tests/golden/fixtures/rank_reconstruction_masks_golden.json
git commit -m "test: lock rank-reconstruction mask builders before Phase 8 move"
```

---

## Task 3: Create `stable_ginv/jacobian/` package + `sweep.py`

**Files:**
- Create: `stable_ginv/jacobian/__init__.py`
- Create: `stable_ginv/jacobian/sweep.py`
- (Leave `functions/jacobian_rank_sweep.py` untouched this task — the monolith still runs; goldens still import the old path and pass.)

- [ ] **Step 1: Create the package marker**

`stable_ginv/jacobian/__init__.py`:

```python
"""Jacobian rank sweep (compute core + CLI), extracted in Phase 8."""
```

- [ ] **Step 2: Create `sweep.py` by relocating the compute core verbatim**

Copy these regions from `functions/jacobian_rank_sweep.py` **verbatim**, in this order, into `stable_ginv/jacobian/sweep.py`:
1. The cache-dir env bootstrap (orig lines 14–20), preceded by the imports it needs: `import sys, os`, `import tempfile`.
2. The remaining worker imports — `import numpy as np`, `import torch`, `import torch.nn as nn`, `import torch.multiprocessing as mp`, `from torchvision import transforms`, `from tqdm import tqdm`.
3. The repointed dependency imports (see "Exact import repointings"): `import functions.consts as consts`, `from functions.Dataset import load_dataset`, `from stable_ginv.io import resolve_storage_paths`, `from helper.Network import get_model, weights_init`, `from stable_ginv.masking import build_gradient_mask`, `from stable_ginv.metrics import compute_jacobian_rank_sweep`.
4. The constant `_TORCH_DTYPES` (orig lines 43–46).
5. Functions, verbatim, in original order: `_parse_explicit_sample_indices` is **NOT** here (it goes to `cli.py`). Include `_get_last_fc_param_indices`, `_split_list`, `_torch_dtype`, `_seed_all`, `_resolve_device`, `_new_rank_results`, `_summarize_rank_results`, `_print_rank_summary`, `_print_mask_debug`, `_worker_core`, `_mp_worker`, `_run_serial`, `_run_parallel`, `_run_dtype`.

Do **not** import matplotlib/seaborn/csv/argparse/datetime here. Do **not** add a `__main__` guard here.

The exact header block of `sweep.py` must be:

```python
import sys, os
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torchvision import transforms

_cache_dir = os.path.join(tempfile.gettempdir(), "stable_ginv_cache")
_mpl_config_dir = os.path.join(_cache_dir, "matplotlib")
_xdg_cache_dir = os.path.join(_cache_dir, "xdg")
os.makedirs(_mpl_config_dir, exist_ok=True)
os.makedirs(_xdg_cache_dir, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _mpl_config_dir)
os.environ.setdefault("XDG_CACHE_HOME", _xdg_cache_dir)

from tqdm import tqdm

import functions.consts as consts
from functions.Dataset import load_dataset
from stable_ginv.io import resolve_storage_paths
from helper.Network import get_model, weights_init
from stable_ginv.masking import build_gradient_mask
from stable_ginv.metrics import compute_jacobian_rank_sweep


_TORCH_DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
}
```

(Then the function bodies follow, copied verbatim.)

- [ ] **Step 3: Verify the module imports and is lint-clean**

Run: `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" && conda run -n stable-ginv python -c "import stable_ginv.jacobian.sweep as s; print(s._summarize_rank_results.__name__, s._mp_worker.__name__, s._run_dtype.__name__)" && conda run -n stable-ginv python -m pyflakes stable_ginv/jacobian/sweep.py stable_ginv/jacobian/__init__.py`
Expected: prints `_summarize_rank_results _mp_worker _run_dtype`; pyflakes reports nothing.

- [ ] **Step 4: Confirm full suite still green (old monolith still in place)**

Run: `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" && conda run -n stable-ginv python -m pytest tests/ -q`
Expected: PASS (goldens still import `functions.jacobian_rank_sweep`, which is unchanged this task).

- [ ] **Step 5: Commit (only when the user authorizes — no co-author trailer)**

```bash
git add stable_ginv/jacobian/__init__.py stable_ginv/jacobian/sweep.py
git commit -m "refactor: extract jacobian sweep compute core into stable_ginv/jacobian/sweep"
```

---

## Task 4: Create `stable_ginv/jacobian/cli.py`

**Files:**
- Create: `stable_ginv/jacobian/cli.py`
- (Leave `functions/jacobian_rank_sweep.py` untouched this task.)

- [ ] **Step 1: Create `cli.py` by relocating `main` + `_parse_explicit_sample_indices` verbatim**

`cli.py` header (note: the `sweep` import comes **before** `import matplotlib` so the cache-dir bootstrap runs first):

```python
import os
import csv
import argparse
from datetime import datetime

import numpy as np
import torch.multiprocessing as mp

from stable_ginv.jacobian.sweep import _run_dtype, _seed_all

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from stable_ginv.io import (
    parse_prefixes_with_fracs,
    resolve_storage_paths,
    safe_makedirs,
    safe_savefig,
)
from functions.Dataset import load_dataset

sns.set_theme(style="whitegrid")
```

Then, **verbatim**:
1. `_parse_explicit_sample_indices` (orig lines 49–59).
2. `main` (orig lines 382–620), unchanged except that it now references the imported `_run_dtype`, `_seed_all`, `parse_prefixes_with_fracs`, `resolve_storage_paths`, `safe_makedirs`, `safe_savefig`, `load_dataset`, `_parse_explicit_sample_indices` — all available via the imports above and the in-module definition. Keep the `import pandas as pd` line **inline inside `main`** exactly where the original has it (orig line 594).

Finally add the `__main__` guard so `python -m stable_ginv.jacobian.cli` works and sets the spawn start method (matching the original `__main__` block, orig lines 623–625):

```python
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
```

- [ ] **Step 2: Verify `--help` output is byte-identical to the original**

Run:
```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python functions/jacobian_rank_sweep.py --help > /tmp/jac_help_old.txt
conda run -n stable-ginv python -m stable_ginv.jacobian.cli --help > /tmp/jac_help_new.txt
diff /tmp/jac_help_old.txt /tmp/jac_help_new.txt && echo "HELP IDENTICAL"
```
Expected: prints `HELP IDENTICAL` (no diff). The original monolith is still present, so this compares old vs new directly.

- [ ] **Step 3: Lint and import-check**

Run: `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" && conda run -n stable-ginv python -m pyflakes stable_ginv/jacobian/cli.py && conda run -n stable-ginv python -c "from stable_ginv.jacobian.cli import main, _parse_explicit_sample_indices; print('ok')"`
Expected: pyflakes silent; prints `ok`.

- [ ] **Step 4: Confirm full suite still green**

Run: `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" && conda run -n stable-ginv python -m pytest tests/ -q`
Expected: PASS.

- [ ] **Step 5: Commit (only when the user authorizes — no co-author trailer)**

```bash
git add stable_ginv/jacobian/cli.py
git commit -m "refactor: extract jacobian sweep CLI into stable_ginv/jacobian/cli"
```

---

## Task 5: Replace `functions/jacobian_rank_sweep.py` with a thin wrapper + repoint its tests

**Files:**
- Modify (replace contents): `functions/jacobian_rank_sweep.py`
- Modify: `tests/test_jacobian_sweep_metadata.py` (repoint import)
- Modify: `tests/golden/test_jacobian_summary_golden.py` (repoint import)

Replacing the monolith breaks the old-path imports in the metadata test and the Task-1 golden, so they are repointed **in this same task** to keep the working tree green.

- [ ] **Step 1: Replace `functions/jacobian_rank_sweep.py` with the wrapper**

```python
"""Thin wrapper: moved to stable_ginv.jacobian (Phase 8).

Kept runnable as `python functions/jacobian_rank_sweep.py ...` and
`python -m functions.jacobian_rank_sweep ...`. The mp.spawn worker now lives at
the stable import path stable_ginv.jacobian.sweep._mp_worker.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch.multiprocessing as mp  # noqa: E402
from stable_ginv.jacobian.cli import main  # noqa: E402

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
```

- [ ] **Step 2: Repoint the metadata test import**

In `tests/test_jacobian_sweep_metadata.py`, change:

```python
from functions.jacobian_rank_sweep import _parse_explicit_sample_indices
```
to:
```python
from stable_ginv.jacobian.cli import _parse_explicit_sample_indices
```

- [ ] **Step 3: Repoint the Task-1 golden import**

In `tests/golden/test_jacobian_summary_golden.py`, change:

```python
from functions.jacobian_rank_sweep import _summarize_rank_results
```
to:
```python
from stable_ginv.jacobian.sweep import _summarize_rank_results
```

- [ ] **Step 4: Verify `--help` still works via the wrapper (both invocation forms)**

Run:
```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python functions/jacobian_rank_sweep.py --help > /tmp/jac_help_wrap.txt
diff /tmp/jac_help_old.txt /tmp/jac_help_wrap.txt && echo "WRAPPER HELP IDENTICAL"
conda run -n stable-ginv python -m functions.jacobian_rank_sweep --help >/dev/null && echo "MODULE FORM OK"
```
Expected: `WRAPPER HELP IDENTICAL` then `MODULE FORM OK`. (`/tmp/jac_help_old.txt` was captured in Task 4 Step 2 from the original; if absent, recapture from git: `git show HEAD~2:functions/jacobian_rank_sweep.py` is the original — or simply confirm the wrapper help matches `python -m stable_ginv.jacobian.cli --help`.)

- [ ] **Step 5: Confirm full suite + goldens green with canonical imports**

Run: `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" && conda run -n stable-ginv python -m pytest tests/ -q && conda run -n stable-ginv python -m pyflakes functions/jacobian_rank_sweep.py`
Expected: PASS; pyflakes silent. The Task-1 golden now proves `_summarize_rank_results` survived the move byte-for-byte.

- [ ] **Step 6: Commit (two categorized commits; only when the user authorizes — no co-author trailer)**

```bash
git add functions/jacobian_rank_sweep.py
git commit -m "refactor: make functions/jacobian_rank_sweep a thin wrapper over stable_ginv.jacobian"
git add tests/test_jacobian_sweep_metadata.py tests/golden/test_jacobian_summary_golden.py
git commit -m "test: import jacobian-sweep tests from canonical stable_ginv.jacobian"
```

---

## Task 6: Move `functions/rank_reconstruction_plot.py` into `stable_ginv/viz/` + wrapper + repoint golden

**Files:**
- Create: `stable_ginv/viz/plot_rank_reconstruction.py`
- Modify (replace contents): `functions/rank_reconstruction_plot.py`
- Modify: `tests/golden/test_rank_reconstruction_masks_golden.py` (repoint import)

- [ ] **Step 1: Create `stable_ginv/viz/plot_rank_reconstruction.py` verbatim from the original**

Copy `functions/rank_reconstruction_plot.py` **whole** into the new file, preserving the module docstring (orig lines 1–19) byte-for-byte (it is the `--help` description via `description=__doc__`). Apply only the import repointings in "Exact import repointings". Do **not** fix the f-string lint yet (Task 7). The new import block (replacing orig lines 30–35) is:

```python
import functions.consts as consts
from functions.Dataset import load_dataset
from stable_ginv.io import resolve_storage_paths
from stable_ginv.masking import _get_last_fc_param_indices, flatten_observed_gradients
from stable_ginv.metrics import compute_grad_match_loss, total_variation, compute_jacobian_rank, _layer_spread_non_fc
from helper.Network import get_model, weights_init
```

Everything else (the `matplotlib.use("Agg")` block, `_build_global_topk_masks`, `_build_keepfc_masks`, `_print_mask_breakdown`, `_reconstruct`, `main`, and the `if __name__ == "__main__": main()` guard) is copied verbatim.

- [ ] **Step 2: Replace `functions/rank_reconstruction_plot.py` with the wrapper**

```python
"""Thin wrapper: moved to stable_ginv.viz.plot_rank_reconstruction (Phase 8).

Kept runnable as `python -m functions.rank_reconstruction_plot ...` and
`python functions/rank_reconstruction_plot.py ...`.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_ginv.viz.plot_rank_reconstruction import main  # noqa: E402

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Repoint the Task-2 golden import**

In `tests/golden/test_rank_reconstruction_masks_golden.py`, change:

```python
from functions.rank_reconstruction_plot import (
    _build_global_topk_masks,
    _build_keepfc_masks,
)
```
to:
```python
from stable_ginv.viz.plot_rank_reconstruction import (
    _build_global_topk_masks,
    _build_keepfc_masks,
)
```

- [ ] **Step 4: Verify `--help` is byte-identical before/after the move**

Run:
```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
git show HEAD:functions/rank_reconstruction_plot.py > /tmp/rr_orig.py
conda run -n stable-ginv python /tmp/rr_orig.py --help > /tmp/rr_help_old.txt
conda run -n stable-ginv python -m functions.rank_reconstruction_plot --help > /tmp/rr_help_new.txt
diff /tmp/rr_help_old.txt /tmp/rr_help_new.txt && echo "RR HELP IDENTICAL"
```
Expected: `RR HELP IDENTICAL`. (`/tmp/rr_orig.py` is the pre-move original from git HEAD; it still imports via its old `functions.*`/`helper.*` paths, which all resolve.)

- [ ] **Step 5: Confirm full suite + goldens green; lint both files**

Run: `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" && conda run -n stable-ginv python -m pytest tests/ -q && conda run -n stable-ginv python -m pyflakes functions/rank_reconstruction_plot.py`
Expected: pytest PASS (the Task-2 golden now proves the mask builders survived byte-for-byte); pyflakes silent on the **wrapper**. Note: `pyflakes stable_ginv/viz/plot_rank_reconstruction.py` will still report the two F541 f-string warnings — that is the pre-existing bug fixed in Task 7.

- [ ] **Step 6: Commit (two categorized commits; only when the user authorizes — no co-author trailer)**

```bash
git add stable_ginv/viz/plot_rank_reconstruction.py functions/rank_reconstruction_plot.py
git commit -m "refactor: move rank-reconstruction plot CLI into stable_ginv/viz with thin wrapper"
git add tests/golden/test_rank_reconstruction_masks_golden.py
git commit -m "test: import rank-reconstruction mask golden from canonical stable_ginv.viz"
```

---

## Task 7: Fix the pre-existing f-string-without-placeholder lint (separate commit)

**Files:**
- Modify: `stable_ginv/viz/plot_rank_reconstruction.py`

The original `functions/rank_reconstruction_plot.py` has two f-strings with no placeholders (orig lines 306 and 310), flagged by pyflakes as F541. The spec lists this as a pre-existing bug to fix **separately** from the move. Removing the `f` prefix produces byte-identical printed output.

- [ ] **Step 1: Remove the spurious `f` prefixes**

In `stable_ginv/viz/plot_rank_reconstruction.py`, change:

```python
        print(f"  entries per layer:")
```
to:
```python
        print("  entries per layer:")
```

and change:

```python
        print(f"  computing Jacobian rank ...")
```
to:
```python
        print("  computing Jacobian rank ...")
```

- [ ] **Step 2: Verify pyflakes is now clean and printed output is unchanged**

Run: `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" && conda run -n stable-ginv python -m pyflakes stable_ginv/viz/plot_rank_reconstruction.py && echo "PYFLAKES CLEAN"`
Expected: prints `PYFLAKES CLEAN` (no F541 warnings). The two edited lines are literal strings with no braces, so `print(...)` output is byte-identical to before.

- [ ] **Step 3: Confirm full suite green**

Run: `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" && conda run -n stable-ginv python -m pytest tests/ -q`
Expected: PASS (mask golden unaffected — the edited lines are diagnostics, not mask logic).

- [ ] **Step 4: Commit (only when the user authorizes — no co-author trailer)**

```bash
git add stable_ginv/viz/plot_rank_reconstruction.py
git commit -m "style: drop placeholderless f-string prefixes in rank-reconstruction plot CLI"
```

---

## Task 8: Update the restructure handover

**Files:**
- Modify: `docs/handover/HANDOVER_RESTRUCTURE.md`

- [ ] **Step 1: Mark Phase 8 complete and add the new goldens**

In the "Golden harness (Phase 0)" list, append two bullets:

```markdown
- `tests/golden/test_jacobian_summary_golden.py` — sweep mean/std summarization
  (added Phase 8, guards the rank-sweep CSV/plot data shaping).
- `tests/golden/test_rank_reconstruction_masks_golden.py` — rank-vs-reconstruction
  mask builders (added Phase 8, guards the global-topk / keep-fc entry selection).
```

In the "Status" list, change the Phase 8 line from `- [ ] Phase 8 …` to:

```markdown
- [x] Phase 8 — `stable_ginv/jacobian/` (sweep.py compute core + cli.py) extracted
      from `functions/jacobian_rank_sweep.py`; `functions/rank_reconstruction_plot.py`
      moved to `stable_ginv/viz/plot_rank_reconstruction.py`. Both `functions/` files
      are thin wrappers. Two goldens added; the placeholderless f-string lint fixed.
```

- [ ] **Step 2: Rewrite the "Next phase" section for Phase 9**

Replace the Phase-8 "Next phase" paragraph with:

```markdown
## Next phase

Phase 9 — Polish: write/expand NumPy-style docstrings across the `stable_ginv/`
package, fill the Sphinx API pages (autosummary), and do a final cleanup pass.
No code moves; no functionality change. Keep this file's Status section current.

Deferred (not yet done): `functions/idlg_cli.py` → `stable_ginv/cli/args.py`,
`functions/Dataset.py`/`functions/consts.py` → `stable_ginv/data/`, and
`helper/Network.py` → `stable_ginv/models/` remain imported from their current
paths (by `stable_ginv/cli/batch.py`, `stable_ginv/recon/runner.py`,
`stable_ginv/viz/plot_model_parameter_counts.py`, `stable_ginv/jacobian/`, and
`stable_ginv/viz/plot_rank_reconstruction.py`).
```

- [ ] **Step 2b: Drop the now-fixed bug from the spec's out-of-scope list (optional, same docs commit)**

In `docs/superpowers/specs/2026-06-04-code-structure-design.md`, the "Out of scope" section lists `functions/rank_reconstruction_plot.py` f-strings missing placeholders as a pre-existing bug. Since Task 7 fixed it, remove that one line (leave the `helper/plots.py:157` `resnet50_data` entry). This keeps the spec accurate.

- [ ] **Step 3: Final full verification**

Run:
```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -m pytest tests/ -q
conda run -n stable-ginv python -m pyflakes stable_ginv functions/jacobian_rank_sweep.py functions/rank_reconstruction_plot.py
conda run -n stable-ginv python iDLG_mask.py --help >/dev/null && echo "iDLG --help OK"
conda run -n stable-ginv python functions/jacobian_rank_sweep.py --help >/dev/null && echo "jac wrapper --help OK"
conda run -n stable-ginv python -m functions.rank_reconstruction_plot --help >/dev/null && echo "rr wrapper --help OK"
git diff --check
```
Expected: pytest PASS; pyflakes silent; three `… OK` lines; `git diff --check` clean.

- [ ] **Step 4: Commit (only when the user authorizes — no co-author trailer)**

```bash
git add docs/handover/HANDOVER_RESTRUCTURE.md docs/superpowers/specs/2026-06-04-code-structure-design.md
git commit -m "docs: mark Phase 8 complete in HANDOVER_RESTRUCTURE"
```

---

## Self-review notes

- **Spec coverage:** Phase 8 row of the spec mapping (`functions/jacobian_rank_sweep.py → stable_ginv/jacobian/sweep.py + CLI`; `functions/rank_reconstruction_plot.py → stable_ginv/viz/`) is implemented by Tasks 3–6. The spec's "Out of scope → pre-existing f-string bug" is resolved separately by Task 7 and reconciled in Task 8. The "Goldens added just-in-time before refactoring an area" invariant is satisfied by Tasks 1–2 preceding the moves. `pyproject.toml` is untouched (editable-install on-disk subpackage), matching the Phase 7 precedent.
- **Multiprocessing correctness:** `_mp_worker` lives in `sweep.py`, a stable importable path, so `mp.spawn` re-imports it cleanly in the child (improvement over spawning from a `__main__`/wrapper module). `mp.set_start_method("spawn", force=True)` is preserved in both the wrapper's and `cli.py`'s `__main__` guards. The cache-dir env bootstrap is at the top of `sweep.py` (the module the worker imports) and runs before any Matplotlib import in `cli.py`.
- **No-change proof:** `--help` byte-diffs (Tasks 4–6) plus the two new goldens (Tasks 1–2, re-run after the move in Tasks 5–6) plus the unchanged full suite enforce identical behavior. The only intentional textual change anywhere is Task 7's `f"…"` → `"…"`, which is output-identical.
- **Type/name consistency:** function names referenced across tasks (`_summarize_rank_results`, `_run_dtype`, `_seed_all`, `_parse_explicit_sample_indices`, `_build_global_topk_masks`, `_build_keepfc_masks`, `main`) match their definitions in the source file read for this plan.
- **Green per task:** every task ends with the working tree passing `pytest tests/ -q`; the only files whose old-path imports break (the metadata test + the two new goldens) are repointed within the same task that introduces the break (Tasks 5–6).
