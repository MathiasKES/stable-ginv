# Current Project Handover

**Last updated:** 2026-06-06

Read this file first when starting a new chat or development session.

## Current State

`stable-ginv` studies how gradient masking affects iDLG image reconstruction.
The active experiment entry point is:

```bash
python iDLG_mask.py --help
```

`iDLG_mask.py` owns CLI parsing, GPU scheduling, registry updates, CSV output,
and high-level orchestration. It spawns workers from `run_single_exp.py`, which
owns reconstruction behavior. Do not execute `run_single_exp.py` directly.

## Read Next

- `HANDOVER_RESEARCHER.md`: architecture, commands, outputs, HPC setup, and
  current behavior.
- `HANDOVER_REVIEWER.md`: invariants and verification rules before editing.
- `HANDOVER_COLLABORATOR.md`: research background and first-run walkthrough.
- `../CODE_CLEANUP.md`: pending conservative cleanup plan.
- `../codebase.md`: detailed function and output reference.

## Current Safety Constraints

- Do not change reconstruction math, masking logic, optimizer behavior, CLI
  defaults, registry key inputs, or CSV columns during cleanup.
- Do not edit modules imported by spawned workers while HPC experiments are
  running. Newly spawned workers read current files from disk.
- Treat `invertinggradients/` as read-only reference code.
- Keep statistical charts in Seaborn. Matplotlib remains appropriate for the
  Agg backend, axes, file saving, and image rendering with `imshow()`.
- Registry-backed analysis scripts should read both legacy and `_v2` registry
  files when possible. Some experiments were appended in multiple batches, so
  paired plotting must align samples by overlapping `run_id` ranges rather than
  assuming both registry entries start at zero.

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
