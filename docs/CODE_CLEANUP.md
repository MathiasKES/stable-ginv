# Code Cleanup Plan

## Goal

Reduce unused code, commented-out code, repeated calculations, and duplicated
helper logic without changing experiment behavior or results.

The cleanup must preserve:

- Reconstruction math and optimizer behavior.
- Masking behavior, including the automatic last-FC-layer inclusion rule.
- CLI flags and defaults.
- Registry key inputs and registry JSON structure.
- CSV filenames, columns, field order, and append behavior.
- Plot filenames and plot contents unless a plotting-only change is explicitly requested.
- Current multiprocessing scheduling and abort-on-worker-failure behavior.

## Active-Run Safety Rule

Do not edit modules imported by `iDLG_mask.py`, `run_single_exp.py`, or spawned
workers while HPC jobs are running. A running parent process can still spawn a
new worker that imports the latest file contents from disk.

Safe while experiments are running:

- Documentation changes.
- Standalone plotting utilities that are not imported by the experiment runner.
- `show_img.py`.
- Removal of comments only.

Wait until experiments finish:

- Any logic change in `functions/`, `helper/Network.py`, `helper/metrics.py`, or
  `helper/visualization.py`.
- Any change to `iDLG_mask.py` or `run_single_exp.py`.
- Module moves, import changes, or function signature changes.

## Completed Standalone Cleanup

These changes are isolated from experiment workers and can be committed
separately:

- `helper/plots.py`
  - Replace comment/uncomment model switching with `--network`.
  - Add `main()` and remove plotting side effects on import.
  - Use the headless Matplotlib `Agg` backend.
  - Derive the layer order from each selected dataset.
  - Preserve Seaborn `FacetGrid` plotting and PDF/PNG outputs.

- `helper/plot_masking_sweep_csv.py`
  - Extract repeated axis formatting and save logic.
  - Avoid repeated registry argument lookups.
  - Warn when sweep rows use different sample counts.
  - Warn when sweep rows map to different baseline configurations.

- `show_img.py`
  - Fail immediately if the requested output directory cannot be created.
  - Always close Matplotlib figures, including after save failures.

- `helper/Network.py`
  - Remove the obsolete commented-out `get_model()` implementation.

## Phase 1: Remove Proven Redundancy — COMPLETED 2026-06-03

Done with regression tests in place first (see below). All changes are
behaviour-preserving; the full suite stays green.

### `functions/Dataset.py`

- Removed the no-op `del imgs, labs` from `_Dataset_from_Image.__init__()`. The
  instance already stores both values via `self.imgs`/`self.labs`, so deleting
  the local names had no effect.

### `helper/visualization.py`

- In `save_restart_curve()`, `_restart_stats()` is now computed once per method
  before the restart-count loop instead of being recomputed for every `k`.
  Restart CSV columns, plot output, and printed summaries are unchanged
  (covered by `tests/test_restart_curve.py`).

### `functions/io_utils.py`

- `_load_registry(path)` and `_save_registry(path, registry)` were already
  shared by the masked and baseline registries.
- Extracted `_registry_key_hash(comparable)` used by `masked_key_from_args()`
  and `baseline_key_from_args()`. The comparable argument dictionaries and JSON
  serialization are unchanged, so existing registry keys are stable (locked by
  `tests/test_registry_keys.py`).

## Phase 2: Optional Dependency Cleanup — COMPLETED 2026-06-03

### `helper/metrics.py`

- `scipy.linalg` is now lazy-loaded via `_load_scipy_linalg()` (called only from
  `_qr_pivot_rows()`) instead of being imported at module top. The loader caches
  the module or the import error, matching the lazy `scipy.stats` pattern in
  `functions/io_utils.py`. The QR pivot calculation and the
  `qr_pivot requires scipy` error message are unchanged
  (`tests/test_metrics_qr_pivot.py`).

Reason: normal reconstruction runs do not require SciPy QR pivoting. Importing
SciPy during worker startup can fail on HPC nodes when the system C++ runtime is
older than the Conda package requirement. (skimage already pulls in scipy core,
but not `scipy.linalg`, so deferring it keeps that submodule out of startup.)

## Phase 3: Standalone Jacobian Sweep Cleanup

`functions/jacobian_rank_sweep.py` is a standalone CLI. Clean it separately from
the main reconstruction runner.

- Reuse `resolve_storage_paths()` from `functions/io_utils.py`.
- Use `safe_makedirs()` and `safe_savefig()` for outputs.
- Replace repeated local HPC storage-path helpers.
- Use `len(sample_indices)` when recording the number of samples.

The last item fixes metadata only: when `--sample_indices` is supplied,
filenames, CSV rows, and legends currently use `args.num_samples` instead of
the actual number of selected indices.

## Phase 4: Larger Organizational Cleanup

Only start this phase after the smaller cleanup is committed and tested.

### Split `functions/io_utils.py`

Consider separating:

- Filesystem, CSV, and storage helpers into `functions/io_utils.py`.
- Statistical calculations into `functions/stats_utils.py`.
- Registry keys and JSON persistence into `functions/registry.py`.

### Split `helper/visualization.py`

Consider separating:

- Reconstruction panels and GIFs into `helper/visualization.py`.
- Restart curves, restart image grids, and restart statistics into
  `helper/restart_visualization.py`.

### Keep Behavior-Sensitive Modules Conservative

Avoid broad rewrites of:

- `functions/masking.py`
- `helper/metrics.py`
- `run_single_exp.py`
- Multiprocessing orchestration in `iDLG_mask.py`

These modules contain distinct branches whose behavior is part of the current
experiments. Extract small helpers only when regression tests cover the existing
behavior.

## Tests To Add Before Larger Refactors

- [x] Registry key stability tests for masked and baseline configurations
  (`tests/test_registry_keys.py`).
- [x] Registry overwrite behavior tests (`tests/test_registry_append.py`).
- [x] Restart-curve CSV regression tests (`tests/test_restart_curve.py`).
- [ ] `merge_restart_results()` and `running_best()` tests.
- [ ] No-plot fallback panel-buffer schema tests.
- [ ] Sweep plot tests covering both `0% baseline` and `0% masked`.
- [ ] Jacobian sweep metadata tests with explicit `--sample_indices`.

Add the remaining items before starting Phases 2–4.

## Verification Checklist

Run after each cleanup commit:

```bash
python -m pyflakes iDLG_mask.py run_single_exp.py functions helper show_img.py tests
python -m py_compile iDLG_mask.py run_single_exp.py functions/io_utils.py \
    functions/experiment_results.py functions/jacobian_rank_sweep.py \
    helper/metrics.py helper/visualization.py helper/plots.py \
    helper/plot_masking_sweep_csv.py show_img.py
python -m pytest tests/ -v
python iDLG_mask.py --help
python show_img.py --help
python helper/plots.py --help
python helper/plot_masking_sweep_csv.py --help
git diff --check
```

When GPU access is available, run a low-cost smoke test:

```bash
python iDLG_mask.py \
    --network resnet18 \
    --dataset cifar10 \
    --num_exp 1 \
    --iteration 10 \
    --num_restarts 1 \
    --methods idlg
```

## Out Of Scope

- `archive/`
- Changes to reconstruction formulas or statistical formulas.
- Changes to experiment defaults.
- Formatting-only churn across the repository.
