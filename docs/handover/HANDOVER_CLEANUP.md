# Handover: Cleanup & Submission Readiness

**Audience:** Anyone continuing repository cleanup or preparing this code for
submission (public GitHub release and/or DTU hand-in).

**Last updated:** 2026-06-03

This file records the *submission-readiness* cleanup — what changed, why, and
what remains. It is distinct from `../CODE_CLEANUP.md`, which is the conservative
*internal code-redundancy* plan (still largely pending; see below).

---

## 1. Status

All work described here lives on the branch `cleanup/submit-ready` and is
**uncommitted** in the working tree. The project rule is no auto-commits, so
review with `git status` and `git diff` before committing. Two untracked items
need `git add` when committing: `CITATION.cff` and `artifacts/figures/`.

---

## 2. Submission context (the decisions that shaped scope)

- **Destination:** both a public GitHub release and a private hand-in to
  examiners. Plan to the stricter public bar, but **do not strip student IDs** —
  the DTU HPC paths and symlinks are kept intentionally.
- **License:** changed from GPLv3 to **MIT** (see §3.3). MIT is permissive and
  the academic norm; citation is handled by `CITATION.cff`, not the license.
- **Generated assets:** consolidated into a single `artifacts/` folder, except
  the HPC symlinks, which stay in place.
- **`invertinggradients/`:** all references removed — it is not on disk, not a
  submodule, and no code imports it (§3.2).

---

## 3. What was done, and why

### 3.1 Consolidated generated assets into `artifacts/`

```
artifacts/
├── plots/    ← was plots/        (ablation forest plots, idx_*.png)
├── figures/  ← was figures/      (illustrations + their generation scripts + lfw_landscape.npz)
├── data/     ← data.json
└── logs/     ← gpuout/*.out (job log) + scripts/results.txt (stale dump)
```

- Tracked files moved with `git mv` to preserve history; the untracked
  `figures/` contents moved with plain `mv`.
- Empty `plots/`, `gpuout/`, and `figures/` directories were removed.
- **Kept in place on purpose:**
  - `hpc/gpuout` and `hpc/results` — symlinks into `/work3/...`; the user wants
    them retained for HPC convenience.
  - `scripts/get_results.txt` — it is a command list consumed by
    `scripts/mean_memory.sh`; moving it would break that script.
- `manual_stats.py` docstring examples were updated to the new
  `artifacts/data/data.json` path.
- `.gitignore` gained `/gpuout/` (off-HPC runtime log fallback written by
  `functions/io_utils.py`) and explicit ignores for agent tooling
  (`.agentic-flow/`, `.swarm/`, `graphify-out/`).

### 3.2 Removed all `invertinggradients` references

The handover docs and README described `invertinggradients/` as a git submodule.
It does not exist on disk, has no `.gitmodules` entry, and nothing imports
`inversefed`/`GradientReconstructor`. References were removed from `README.md`
and all docs, including the `--recurse-submodules` clone command. The methods
from that paper are implemented directly in this repo (§3.6), so no reference
code is needed.

### 3.3 License, citation, and metadata

- `LICENSE`: GPLv3 → **MIT**, copyright holders Alfred Aqraou, Ali Afif,
  Mathias Sørensen.
- Added `CITATION.cff` and README **License** + **Citation** sections. The
  citation is the thesis only — the underlying papers are credited by
  author/year in prose, not as a formal reference list (deliberate, per author
  instruction).
- Thesis title set to **"Stabilizing Gradient Inversion in Federated Learning"**
  (from the thesis cover page) in both the README and `CITATION.cff`.

### 3.4 Environment files

- README and docs pointed at a non-existent root `environment.yml`; corrected to
  `env/environment.yml`.
- Renamed `env/environment cpu.yml` → `env/environment-cpu.yml` (the space in the
  filename was fragile).
- Added `pandas` as an explicit dependency in both env files (it was only present
  transitively via seaborn, and code now imports it directly — §3.5), and removed
  a duplicate `seaborn` line.

### 3.5 Seaborn 0.13 compatibility fix

`save_restart_curve()` in `helper/visualization.py` passed a **list of dicts** to
`sns.lineplot(data=...)`. Seaborn ≥0.13 rejects lists for `data=` and requires a
DataFrame/Mapping; older seaborn (on HPC) accepted lists, so this only failed
locally. Fixed by wrapping in `pd.DataFrame(line_rows)` (identical plot output).
This was the only list-to-`data=` occurrence in the repo — all other seaborn
calls already use DataFrames or `x=`/`y=` vectors.

### 3.6 Documentation accuracy

- Corrected the project framing from "iDLG only" to the full attack family it
  implements: **DLG** (Zhu et al., 2019), **iDLG** (Zhao et al., 2020), and
  **Inverting Gradients** (Geiping et al., 2020). The reconstruction loop uses
  Geiping's cosine-similarity matching with a TV prior plus the iDLG
  label-recovery trick. Updated in README, `HANDOVER_RESEARCHER.md`, and
  `HANDOVER_COLLABORATOR.md`.

### 3.7 CPU execution support (additive)

The runner was CUDA-only. CPU support was added as a strict fallback — **GPU
behavior is unchanged**, CPU is used only when `torch.cuda.device_count() == 0`:

- `run_single_exp.py`: device is `cuda:{device_id}` when CUDA is available (as
  before), else `'cpu'`.
- `iDLG_mask.py`: the hard `raise RuntimeError("No CUDA GPUs available.")` became
  a CPU fallback that runs a **single** worker (`num_workers = 1`); scheduling
  loops iterate over `num_workers` instead of `num_gpus`; the end-of-run
  `torch.cuda.memory.*` prints are guarded by `torch.cuda.is_available()`.
- Rationale for a single CPU worker: multiple heavy CPU torch processes would
  oversubscribe cores. `torch.cuda.manual_seed_all(...)` is a documented no-op
  with no CUDA, so CPU runs stay seeded via `torch.manual_seed`.
- README, `HANDOVER_RESEARCHER.md`, and `HANDOVER_REVIEWER.md` were updated to
  describe the slow-but-functional CPU fallback. CPU is for small test runs
  only; a GPU is strongly recommended.

---

## 4. Invariants preserved

No change touched any of the following (per `CLAUDE.md` and
`HANDOVER_REVIEWER.md`): reconstruction math, masking logic, optimizer behavior,
CLI flags/defaults, registry key inputs, registry JSON structure, CSV
filenames/columns/order, or plot contents. Statistical charts remain in Seaborn.

---

## 5. How to verify cleanup changes

```bash
# Run inside the project conda env; on DTU HPC also export the library path first.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"   # HPC only

conda run -n stable-ginv python -m py_compile iDLG_mask.py run_single_exp.py \
    functions/*.py helper/*.py show_img.py manual_stats.py
conda run -n stable-ginv python iDLG_mask.py --help
conda run -n stable-ginv python show_img.py --help
git diff --check
```

Environment caveats discovered this session:

- **`pyflakes` and `pytest` are not installed** in the `stable-ginv` env, so the
  `pyflakes`/`pytest` lines in the older verification blocks do not run as-is.
  Use `py_compile` + `--help` (which import the modules) as the lightweight
  check, or `pip install pyflakes pytest` first to run the full suite.
- **Do not run a real experiment locally to "test."** `resolve_storage_paths()`
  resolves to `/work3/s234843/bachelor/results` whenever that path is writable —
  which it is on the author's local machine via a symlink — so a live run
  appends to the **real** registry/CSV. To smoke-test the pipeline safely,
  monkeypatch `resolve_storage_paths` to a temp dir, or run where `/work3` is not
  accessible.
- After any code edit, run `graphify update .` (AST-only, no API cost) to keep
  `graphify-out/` current.

---

## 6. Pending / future cleanup

- **Internal code-redundancy refactors** in `../CODE_CLEANUP.md`: **Phases 1, 2,
  and 3 are done** (2026-06-03) with new regression tests
  (`tests/test_registry_keys.py`, `tests/test_restart_curve.py`,
  `tests/test_metrics_qr_pivot.py`, `tests/test_jacobian_sweep_metadata.py`); the
  full suite is green (78 tests). **Phase 4 (module splits) remains** — do it only
  after active HPC runs finish, and add the remaining tests listed in that doc
  first.
- A local CPU smoke run during this work wrote test rows into the **real**
  `/work3/.../results` store. **Prune these on the HPC** — the `/work3` mount on
  the author's local machine is a *stale* copy; the live results store is
  HPC-only. The contamination has a clean signature (all from the
  `LeNet`/`MNIST`/`num_exp 2`/`num_restarts 2` quick-test in
  `scripts/cmds.txt`, run as user `mathias`, dated `20260603`):
  - `exp_results_LeNet.csv` — one row with `Run by=mathias`, empty
    `job_id`/`device`, and an extra trailing registry-hash column no other row
    has (key `f2bcf2cb…`).
  - `baselines/idlg_baselines_summary_v2.csv` — the `f2bcf2cb…` row
    (`num_exp 2`); keep the real `d16e59d3…` baseline (`num_exp 100`).
  - `baselines/idlg_baselines_registry_v2.json` — drop key `f2bcf2cb…`.
  - Standalone test artifacts: `20260603_*_0.png`,
    `restart_curve_20260603_*.{csv,png}`, `restart_images_20260603_*.png`.
- Commit the `cleanup/submit-ready` branch once reviewed.

Resolved since the prior revision of this section:

- **`prefix:` line** removed from both `env/*.yml` (was a hardcoded HPC path that
  warned on other machines; `conda env create -f` works by name without it).
- **CUDA version docs** reconciled to **CUDA 12.6** to match the `cu126` PyTorch
  wheel (`HANDOVER_COLLABORATOR.md` previously said 12.8).

---

## 7. Pitfalls specific to cleanup

- Spawned workers re-read modules from disk (`spawn` start method). Do not edit
  modules imported by `iDLG_mask.py` / `run_single_exp.py` while HPC jobs run.
- `artifacts/figures/` is untracked until `git add`; it will not appear in a
  fresh clone otherwise.
- Keep handover files current-state only — record history in Git, not in the
  active docs.
