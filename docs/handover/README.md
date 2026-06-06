# Current Project Handover

**Last updated:** 2026-06-06

Read this file first when starting a new chat or development session.

## Current State

`stable-ginv` studies how gradient masking affects gradient-inversion image
reconstruction (DLG / iDLG / Inverting Gradients). The active experiment entry
point is:

```bash
python iDLG_mask.py --help
```

`iDLG_mask.py` owns CLI parsing, GPU/CPU scheduling, registry updates, CSV
output, and high-level orchestration. It spawns workers from `run_single_exp.py`,
which owns reconstruction behavior. Do not execute `run_single_exp.py` directly.
A GPU is strongly recommended; with no GPU the runner falls back to a single,
much slower CPU worker.

## Read Next

- `HANDOVER_RESEARCHER.md`: architecture, commands, outputs, HPC setup, and
  current behavior.
- `HANDOVER_REVIEWER.md`: invariants and verification rules before editing.
- `HANDOVER_COLLABORATOR.md`: research background and first-run walkthrough.
- `HANDOVER_CLEANUP.md`: submission-readiness cleanup — what was done, the
  decisions behind it, how to verify, and what remains.
- `../CODE_CLEANUP.md`: pending conservative internal code-redundancy plan.
- `HANDOVER_RESTRUCTURE.md`: OOP restructure status, invariants, and golden
  harness. Read before any restructure work.
- `../codebase.md`: detailed function and output reference.

## Current Safety Constraints

- Do not change reconstruction math, masking logic, optimizer behavior, CLI
  defaults, registry key inputs, or CSV columns during cleanup.
- Do not edit modules imported by spawned workers while HPC experiments are
  running. Newly spawned workers read current files from disk.
- Keep statistical charts in Seaborn. Matplotlib remains appropriate for the
  Agg backend, axes, file saving, and image rendering with `imshow()`.

## Handover Maintenance Rule

Whenever handover files are created or updated:

- Keep them current-state only. Remove redundant, outdated, or superseded
  instructions instead of appending historical notes.
- Ensure they contain everything a new agent needs to understand the active
  architecture, run the current commands, avoid known pitfalls, verify changes,
  and continue pending work.
- Keep historical details in Git history, not in the active handover files,
  unless they directly affect interpretation of existing experiment results.
- Update this starting file when the recommended reading order or active entry
  points change.

## DTU HPC Requirement

After activating Conda and before running Python:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

Without this, SciPy or Matplotlib may load the system C++ runtime and fail with
`CXXABI_1.3.15 not found`.

## Basic Verification

```bash
python -m pyflakes iDLG_mask.py run_single_exp.py functions helper show_img.py tests
python -m pytest tests/ -v
python iDLG_mask.py --help
git diff --check
```

`pyflakes` and `pytest` may not be installed in the env — if so, either
`pip install pyflakes pytest` or use the `py_compile` + `--help` fallback in
`HANDOVER_CLEANUP.md` §5. Do not run a real experiment locally to test: outputs
resolve to the real `/work3/.../results` (see `HANDOVER_CLEANUP.md` §5).
