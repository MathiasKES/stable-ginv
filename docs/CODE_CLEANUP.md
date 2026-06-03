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

## Phase 1: Remove Proven Redundancy

Perform this phase after active HPC runs finish.

### `functions/Dataset.py`

- Remove `del imgs, labs` from `_Dataset_from_Image.__init__()`. The instance
  already stores both values, so deleting local variables has no effect.

### `helper/visualization.py`

- In `save_restart_curve()`, calculate `_restart_stats()` once per method before
  the restart-count loop. The current implementation recalculates identical
  statistics for every `k`.
- Preserve restart CSV columns, restart plot output, and printed summaries.

### `functions/io_utils.py`

- Extract an internal `_load_registry(path)` helper used by masked and baseline
  registries.
- Extract an internal `_save_registry(path, registry)` helper used by masked and
  baseline registries.
- Extract an internal hashing helper used by `masked_key_from_args()` and
  `baseline_key_from_args()`.
- Preserve the exact comparable argument dictionaries and JSON serialization so
  existing registry keys remain unchanged.

## Phase 2: Optional Dependency Cleanup

Perform this phase after active HPC runs finish.

### `helper/metrics.py`

- Lazy-load `scipy.linalg` only when QR pivoting is requested.
- Cache the imported module or import error, matching the lazy `scipy.stats`
  pattern in `functions/io_utils.py`.
- Preserve the existing QR pivot calculation and error message behavior.

Reason: normal reconstruction runs do not require SciPy QR pivoting. Importing
SciPy during worker startup can fail on HPC nodes when the system C++ runtime is
older than the Conda package requirement.

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

- Registry key stability tests for masked and baseline configurations.
- Registry overwrite behavior tests.
- Restart-curve CSV regression tests.
- `merge_restart_results()` and `running_best()` tests.
- No-plot fallback panel-buffer schema tests.
- Sweep plot tests covering both `0% baseline` and `0% masked`.
- Jacobian sweep metadata tests with explicit `--sample_indices`.

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
