# Handover: OOP Restructure

**Spec:** `docs/superpowers/specs/2026-06-04-code-structure-design.md`
**Plans:** `docs/superpowers/plans/`
**Branch:** `cleanup/submit-ready`

## Invariants (every phase)

- No functionality change: reconstruction math, masking behavior, optimizer
  behavior, CLI flags/defaults, registry key inputs/JSON, CSV
  filenames/columns/order/append, plot filenames/contents stay identical.
- Repo stays green and runnable after every phase.
- Categorized commits: `refactor:` / `test:` / `docs:` / `build:` / `style:`
  (no `feat:`). Separate commit per category.
- Goldens are added just-in-time: before refactoring an area, lock it with a
  golden test.

## Verify (run after any change)

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -m pytest tests/ -q
conda run -n stable-ginv python iDLG_mask.py --help
git diff --check
```

Regenerate goldens only when the env (e.g. torch version) changes, never to make
a refactor pass:
```bash
GOLDEN_REGEN=1 conda run -n stable-ginv python -m pytest tests/golden/ -q
```

## Golden harness (Phase 0)

- `tests/golden/test_masking_golden.py` — all mask modes (keep_ids/entry_masks).
- `tests/golden/test_registry_key_golden.py` — masked keys across modes.
- `tests/golden/test_recon_worker_golden.py` — in-process recon numerics (CPU).
- `tests/golden/test_baseline_summary_csv_golden.py` — baseline-summary CSV bytes
  (added Phase 4, guards the registry-summary writer).
- Helpers + fixtures: `tests/golden/helpers.py`, `tests/golden/fixtures/`.

Note: the spec's "end-to-end golden" is realized in-process (recon-worker golden)
because `resolve_storage_paths()` resolves to the shared `/work3` tree locally, so
a full-CLI run would append to real experiment data. The baseline-summary CSV golden
(`tests/golden/test_baseline_summary_csv_golden.py`) was added in Phase 4 to guard the
registry-summary writer; the per-run `experiment_results` CSV-row golden is added in
Phase 6, just before that refactor.

## Status

- [x] Phase 0 — Scaffolding: package skeleton, `pyproject.toml`, golden harness,
      Sphinx + Pages CI, this handover.
- [x] Phase 1 — `ExperimentConfig` dataclass replaces the config dict.
- [x] Phase 2 — `stable_ginv/metrics/`.
- [x] Phase 3 — `stable_ginv/masking/` strategy classes.
- [x] Phase 4 — `io_utils` teardown → `registry/` + `stats/` + `io/`.
- [x] Phase 5 — `stable_ginv/recon/` (ReconstructionRunner, EarlyStopPolicy,
      LabelInference, scheduler). `run_single_exp.py` + `helper/training_utils.py`
      are shims.
- [ ] Phase 6 — `stable_ginv/experiment/` (+ CSV-row goldens first).
- [ ] Phase 7 — `stable_ginv/viz/`.
- [ ] Phase 8 — `stable_ginv/jacobian/`.
- [ ] Phase 9 — docstrings + fill Sphinx API pages + polish.

## One-time setup

- Enable GitHub Pages: repo Settings → Pages → Source = "GitHub Actions".

## Next phase

Write the Phase 6 plan from the spec, then implement. Phase 6 extracts
`stable_ginv/experiment/` — `BatchExperimentRunner` (GPU scheduling, multiprocessing,
output ordering, abort-on-worker-failure), `ResultAggregator`, and `RestartSelector`,
plus `functions/experiment_results.py` → `stable_ginv/experiment/results.py` +
`stable_ginv/io/csv.py`. **Add the per-run `experiment_results` CSV-row golden first**
(the spec defers it to just before this refactor), then move `iDLG_mask.py`
orchestration into `stable_ginv/cli/batch.py` + `stable_ginv/experiment/runner.py`,
leaving `iDLG_mask.py` as a thin `from stable_ginv.cli.batch import main; main()`
wrapper. Keep this file's Status section current at every phase boundary.
