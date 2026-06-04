# Code Structure Redesign — Design Spec

**Date:** 2026-06-04
**Status:** Approved design, pending implementation plan
**Branch:** `cleanup/submit-ready`

## Goal

Restructure `stable-ginv` (~9,300 lines of mostly procedural Python) into a
readable, object-oriented, importable package **without changing any
functionality**. The end state must remain easy to interpret and run.

### Hard constraints

- **No functionality change, ever.** Reconstruction math, masking behavior,
  optimizer behavior, CLI flags/defaults, registry key inputs, registry JSON
  structure, CSV filenames/columns/order/append behavior, and plot
  filenames/contents stay identical.
- **Identical CLI & on-disk outputs.** Existing HPC commands
  (`python iDLG_mask.py ...`) and existing result registries keep working
  untouched. The refactor is internal-only.
- **Readability is the primary objective:** OOP where it clarifies, small
  focused modules (target ≤ ~300 lines/file), idiomatic names.
- **Phased and resumable.** Repo stays green and runnable after every phase. A
  living handover doc lets a fresh session resume cold.

## Decisions (locked)

| Question | Decision |
|----------|----------|
| Restructure depth | Deep OOP redesign |
| Packaging | Real `stable_ginv/` package + thin root wrappers (`pip install -e .`) |
| No-change proof | Golden characterization tests (assert `diff == 0`) |
| Docs tool | **Sphinx** (autodoc + napoleon) → GitHub Pages via Actions. Supersedes the original Doxygen request. |
| Back-compat | Identical CLI & outputs |
| Migration style | Incremental strangler, phased |
| Branch | Continue on `cleanup/submit-ready` (merge to main handled separately by user) |
| Renaming | Rename internals freely; outputs/CLI unchanged |
| Masking OOP | Strategy class per mode + registry |

## Section 1 — Target package architecture

```
stable_ginv/
├── config.py          ExperimentConfig (frozen, picklable dataclass) — replaces UPPERCASE config dict
├── cli/               argparse definitions + main() per entry point; builds ExperimentConfig
├── experiment/        BatchExperimentRunner (scheduling, multiprocessing, output ordering)
│                      ResultAggregator, RestartSelector
├── recon/             ReconstructionRunner, EarlyStopPolicy, LabelInference, scheduler
├── masking/           MaskStrategy protocol + one class per mode + STRATEGY_REGISTRY + Masker facade
├── metrics/           image_metrics (psnr/ssim/tv), jacobian, grad_match
├── registry/          MaskedRegistry, BaselineRegistry, key hashing
├── stats/             paired comparisons, confidence intervals (lazy scipy)
├── io/                StoragePaths, CSV writers, safe_makedirs/safe_savefig
├── data/              dataset loaders + normalization constants
├── models/            model factory + custom CNNs
├── viz/               recon panels/GIFs, restart curves, standalone plot CLIs
└── jacobian/          rank sweep + its CLI

iDLG_mask.py            -> from stable_ginv.cli.batch import main; main()
run_single_exp.py       -> worker shim (kept importable for multiprocessing spawn)
show_img.py             -> thin wrapper
manual_stats.py         -> thin wrapper
helper/plot_*.py        -> thin wrappers (kept until viz phase, then wrappers)
pyproject.toml          -> editable install + console entry points
```

### Current → target module mapping

| Current | Target |
|---------|--------|
| `iDLG_mask.py` (orchestration) | `stable_ginv/cli/batch.py` + `stable_ginv/experiment/runner.py` |
| `run_single_exp.py` (worker) | `stable_ginv/recon/runner.py` (+ root worker shim) |
| `functions/idlg_cli.py` | `stable_ginv/cli/args.py` |
| `functions/masking.py` | `stable_ginv/masking/` (strategies + registry + facade) |
| `functions/experiment_results.py` | `stable_ginv/experiment/results.py` + `stable_ginv/io/csv.py` |
| `functions/io_utils.py` (732 lines) | split → `stable_ginv/registry/`, `stable_ginv/stats/`, `stable_ginv/io/` |
| `functions/Dataset.py`, `functions/consts.py` | `stable_ginv/data/` |
| `functions/jacobian_rank_sweep.py` | `stable_ginv/jacobian/sweep.py` + CLI |
| `functions/rank_reconstruction_plot.py` | `stable_ginv/viz/` (rank-vs-reconstruction plot CLI) |
| `helper/Network.py` | `stable_ginv/models/` |
| `helper/metrics.py` (677 lines) | `stable_ginv/metrics/` (image, jacobian, grad_match) |
| `helper/visualization.py` (620 lines) | `stable_ginv/viz/` (panels, restart) |
| `helper/training_utils.py` | `stable_ginv/recon/scheduler.py` |
| `helper/plot_*.py` | `stable_ginv/viz/` standalone CLIs |
| `show_img.py`, `manual_stats.py` | `stable_ginv/cli/` (+ root wrappers) |

### Key OOP abstractions

- **`ExperimentConfig`** — frozen dataclass replacing the `NETWORK_NAME`/`Iteration`/…
  dict. Picklable so it crosses the multiprocessing worker boundary cleanly.
- **`MaskStrategy`** protocol + one strategy class per mode (`GradsizeTopK`,
  `GradsizeTopFracEntriesLayer`, `PrefixTopFracEntriesLayer`, …) +
  `STRATEGY_REGISTRY` keyed by mode name + a `Masker` facade. Must return
  identical `keep_ids`/`entry_masks` (guarded by masking goldens).
- **`ReconstructionRunner`** — owns the optimization loop; `EarlyStopPolicy`
  dataclass; `LabelInference` (one-time, from original unmasked FC gradient).
- **`MaskedRegistry` / `BaselineRegistry`** — load/save/key; preserve exact
  comparable-args dicts and JSON serialization (incl. `fc_forced` marker).
- **`ResultAggregator` / `RestartSelector`** — aggregation, paired summaries,
  CSV row construction.
- **`BatchExperimentRunner`** — GPU scheduling, multiprocessing, output ordering,
  abort-on-worker-failure behavior (unchanged).
- **`StoragePaths`** — HPC/local path resolution.

The multiprocessing subtlety: `run_single_exp.py` must remain importable at a
stable path for worker spawn; the worker entry stays as a shim that delegates to
`stable_ginv.recon`.

## Section 2 — Golden characterization harness

Built in **Phase 0, before any refactor**. All assert `diff == 0`.

1. **Registry-key goldens** — every mask/baseline config → exact key hash.
   Extends existing `tests/test_registry_keys.py`.
2. **Masking goldens** — fixed small net + fixed gradients; run **all ~15 modes**;
   hash resulting `keep_ids`/`entry_masks`; assert identical. Directly guards the
   strategy-class extraction (Phase 3).
3. **End-to-end goldens** — one seeded CPU run (LeNet/MNIST,
   `num_exp=1 iteration=10 num_restarts=1`) captures CSV columns+values, registry
   JSON, and recon metrics into `tests/golden/`. Asserted identical after each
   phase. Exact equality is the target; switch to a tight tolerance only if
   floating-point op-order proves unstable on CPU.

Run in CI and locally after every phase. A drifted number turns the build red.

## Section 3 — Phase plan

Each phase ends green (pytest + goldens pass, `--help` works), with categorized
commits and an updated handover doc.

- **Phase 0 — Scaffolding:** `stable_ginv/` skeleton, `pyproject.toml` (editable
  install), golden harness, Sphinx + GitHub Pages CI, `HANDOVER_RESTRUCTURE.md`.
  No code moves yet.
- **Phase 1 — Config:** `ExperimentConfig` dataclass replaces the config dict in
  `iDLG_mask.py` / `run_single_exp.py`; outputs identical.
- **Phase 2 — Metrics:** extract `stable_ginv/metrics/`.
- **Phase 3 — Masking:** strategy classes + registry + facade.
- **Phase 4 — io_utils teardown:** `registry/` + `stats/` + `io/`.
- **Phase 5 — Reconstruction:** `recon/` runner OOP.
- **Phase 6 — Orchestration:** `experiment/` runner + result aggregation.
- **Phase 7 — Visualization:** `viz/` split.
- **Phase 8 — Jacobian:** sweep + standalone plot CLIs.
- **Phase 9 — Polish:** docstring pass, fill Sphinx API pages, final cleanup.

## Section 4 — Documentation (Sphinx → GitHub Pages)

- `docs/` Sphinx project: `autodoc` + `napoleon` (NumPy/Google docstrings) +
  `autosummary`, ReadTheDocs theme.
- GitHub Actions workflow builds on push to the working branch and deploys to
  GitHub Pages.
- NumPy-style docstrings written/expanded during each phase migration; exhaustive
  pass in Phase 9.

## Section 5 — Commits, branch, handover

- **Branch:** `cleanup/submit-ready`.
- **Categorized commits** (no `feat:`, since functionality never changes):
  `refactor:` (code moves/OOP), `test:` (goldens), `docs:` (docstrings/Sphinx),
  `build:` (pyproject/CI), `style:` (renames). Separate commit per category within
  a phase.
- **Handover:** `docs/handover/HANDOVER_RESTRUCTURE.md` — current phase, done,
  next, how to verify, invariants. Updated at every phase boundary; wired into
  `docs/handover/README.md` reading order.

## Verification (every phase)

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -m pytest tests/ -q          # incl. goldens
conda run -n stable-ginv python -m pyflakes stable_ginv ...  # lint
conda run -n stable-ginv python iDLG_mask.py --help
git diff --check
```

## Out of scope

- `archive/`, `invertinggradients/` (read-only reference).
- Any change to reconstruction/statistical formulas or experiment defaults.
- Merge to `main` (handled separately by the user).
- Fixing pre-existing bugs unrelated to a phase (tracked separately):
  `helper/plots.py:157` undefined `resnet50_data`;
  `functions/rank_reconstruction_plot.py` f-strings missing placeholders.
