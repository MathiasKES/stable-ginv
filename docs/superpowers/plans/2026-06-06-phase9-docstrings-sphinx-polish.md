# Phase 9 — Docstrings, Sphinx API Pages & Final Polish — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the OOP restructure with NumPy-style docstrings across the entire `stable_ginv/` package, a clean (warning-free) Sphinx build that renders every subpackage, and a final verification/handover pass — with zero functionality change.

**Architecture:** Phase 9 is documentation-and-polish only. **No code moves, no behavior change.** Three kinds of edits: (1) Sphinx infra so the recursive autosummary imports every module under mocked heavy deps and builds warning-free; (2) docstrings added to the ~48 public symbols / ~17 modules that currently lack them; (3) handover/status updates. The `docs/sphinx/api/*.rst` pages are **git-ignored and auto-generated** by `autosummary :recursive:` at build time — never hand-write or commit them.

**Tech Stack:** Python 3.13 (conda env `stable-ginv`), Sphinx 7+ (autodoc + autosummary + napoleon, RTD theme), pytest goldens, pyflakes.

---

## Invariants (carry from `docs/handover/HANDOVER_RESTRUCTURE.md`)

- **No functionality change.** Reconstruction math, masking, optimizer, CLI flags/defaults, registry keys/JSON, CSV columns, plot filenames/contents stay identical.
- **`--help` text is an output.** Two viz CLIs pass their module docstring straight to argparse: `stable_ginv/viz/plot_rank_reconstruction.py:212` and `stable_ginv/viz/plot_model_parameter_counts.py:114` both use `description=__doc__`. **Do NOT change the text of those two module docstrings** — it would change `--help` output. (Their *function* docstrings are safe to add.) Every other module docstring is free to edit.
- Repo stays green and runnable after every task.
- Categorized commits, **separate commit per category**: `docs:` (docstrings/Sphinx prose), `build:` (conf.py / CI / pyproject). No `feat:`. No `refactor:`/`style:`/`test:` expected this phase (no code moves, no new tests).
- Goldens are not regenerated to make anything pass.

## Baseline (already verified at plan-writing time)

- `pytest tests/ -q` → **117 passed**.
- `pyflakes stable_ginv` → clean.
- Current docs build → "build succeeded, 2 warnings": `stable_ginv.viz` **fails to import** (so no viz pages render) because viz modules `import pandas as pd` and `pandas` is **not** in `autodoc_mock_imports`; under the docs CI (`pip install -e . --no-deps`) real pandas parses the mocked numpy's `__version__` and raises `TypeError: expected string or bytes-like object, got '__version__'`.
- After Task 1's three fixes, the build is **clean even with `-W`** (verified).

## Verification commands (used throughout)

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -m pytest tests/ -q
conda run -n stable-ginv python -m pyflakes stable_ginv
conda run -n stable-ginv python iDLG_mask.py --help
conda run -n stable-ginv python -m sphinx -W -b html docs/sphinx /tmp/sphinx_check
git diff --check
```

**Docstring-coverage checker** (save once at repo root as `scripts/_docstring_audit.py`; it is a throwaway dev tool — delete it in Task 8, do not commit it). Run it scoped to a subpackage to confirm a task is complete:

```python
# scripts/_docstring_audit.py  — usage: python scripts/_docstring_audit.py stable_ginv/masking
import ast, glob, sys
root = sys.argv[1] if len(sys.argv) > 1 else "stable_ginv"
missing = []
for f in sorted(glob.glob(f"{root}/**/*.py", recursive=True)) + sorted(glob.glob(f"{root}.py")):
    tree = ast.parse(open(f).read())
    if ast.get_docstring(tree) is None:
        missing.append(f"{f}: <module>")
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            n = node.name
            if n.startswith("_") and not (n.startswith("__") and n.endswith("__")):
                continue
            if ast.get_docstring(node) is None:
                missing.append(f"{f}: {n}")
print("\n".join(missing) if missing else f"OK: all public symbols in {root} documented")
```

> Note: nested local functions named `closure` (in `recon/runner.py` and `viz/plot_rank_reconstruction.py`) are flagged by the auditor but are *local* — autodoc never renders them. Give each a one-line docstring anyway for cleanliness; that satisfies the auditor.

## Docstring style (NumPy / napoleon)

Match the existing house style (see `stable_ginv/config.py`, `stable_ginv/metrics/image_metrics.py`, `stable_ginv/recon/early_stop.py`). Rules:

- One-line imperative summary, then a blank line, then detail if needed.
- Document real `Parameters`/`Returns`/`Raises` only when they aid understanding; short pure helpers may stay one-line. **Read the function body before writing** — describe what it actually does; never invent parameters or behavior.
- Keep lines ≤ ~88 cols. Use double-backticks for code/identifiers in prose.
- Do not add type annotations as prose if they're already in the signature (`autodoc_typehints = "description"` renders them).

Worked example (the target shape):

```python
def compute_psnr_from_mse(mse, max_val=1.0):
    """Convert a mean-squared-error value to PSNR in decibels.

    Parameters
    ----------
    mse : float
        Mean squared error between two images.
    max_val : float, optional
        Maximum possible pixel value (1.0 for normalized images).

    Returns
    -------
    float
        Peak signal-to-noise ratio in dB, or ``inf`` when ``mse == 0``.
    """
```

---

## File Structure

No files are created except the throwaway auditor (deleted in Task 8). Files modified:

- **Task 1 (build):** `docs/sphinx/conf.py`, `.github/workflows/docs.yml`; **(docs):** `stable_ginv/viz/plot_masking_sweep_csv.py` (module docstring only).
- **Tasks 2–7 (docs):** docstrings inside `stable_ginv/**` modules (exact symbols listed per task).
- **Task 8 (docs):** `docs/handover/HANDOVER_RESTRUCTURE.md`, `docs/handover/README.md`.

---

### Task 1: Sphinx build hardening (render all subpackages, warning-free)

**Files:**
- Modify: `docs/sphinx/conf.py:26-27` (mock list) and append `rst_prolog`
- Modify: `.github/workflows/docs.yml:31` (build with `-W`)
- Modify: `stable_ginv/viz/plot_masking_sweep_csv.py:1-17` (module docstring — this module uses a plain `ArgumentParser()`, so its docstring is **not** `--help` text and is safe to edit)

- [ ] **Step 1: Add `pandas` to the autodoc mock list**

In `docs/sphinx/conf.py`, change:

```python
autodoc_mock_imports = ["torch", "torchvision", "scipy", "skimage", "seaborn",
                        "matplotlib", "numpy", "imageio", "PIL", "tqdm"]
```
to:
```python
autodoc_mock_imports = ["torch", "torchvision", "scipy", "skimage", "seaborn",
                        "matplotlib", "numpy", "imageio", "PIL", "tqdm", "pandas"]
```

- [ ] **Step 2: Define the `|grad|` substitution so the rank-reconstruction help-text docstring renders**

`stable_ginv/viz/plot_rank_reconstruction.py`'s module docstring contains a literal `|grad|` and is reused verbatim as argparse `--help` text, so it must not be edited. Define the substitution in Sphinx instead. Append to the end of `docs/sphinx/conf.py`:

```python

# stable_ginv/viz/plot_rank_reconstruction.py reuses its module docstring verbatim
# as argparse --help text (description=__doc__), so the literal "|grad|" in it must
# stay. Define the substitution here so docutils renders it literally instead of
# erroring on an undefined substitution.
rst_prolog = r".. |grad| replace:: \|grad\|"
```

- [ ] **Step 3: Fix the `plot_masking_sweep_csv.py` module docstring indentation (the only remaining ReST error)**

Replace the entire current module docstring (lines 1–17, the first `"""..."""` block) with this reST-clean version. Content is preserved; indented shell snippets become proper literal blocks (`::`):

```python
"""Plot masking-sweep results from sweep CSV rows that reference masked entries.

The sweep CSV is produced by::

    python iDLG_mask.py --methods masked \\
        --mask_mode gradsize_topfrac_entries_layer --gradsize_topfrac 0.5

Example::

    python helper/plot_masking_sweep_csv.py \\
        results/masking_sweeps/mse_resnet18_cifar100_gradsize_topfrac_entries_layer_<hash>.csv \\
        --out_dir results/masking_sweep_plots

Default threshold is ``--threshold_mse 0.01``. The summary CSV includes
network/dataset columns, and plot titles show both. Output filenames include
network, dataset, and threshold, e.g. ``sweep_plot_vgg13_cifar100_threshold_0p01.png``.
"""
```

- [ ] **Step 4: Build the docs with `-W` and confirm zero warnings + viz pages present**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
rm -rf docs/sphinx/api /tmp/sphinx_check
conda run -n stable-ginv python -m sphinx -W -b html docs/sphinx /tmp/sphinx_check 2>&1 | tail -5
ls docs/sphinx/api/ | grep viz   # expect 10 viz.* pages
```
Expected: `build succeeded` (no "N warnings"), and viz pages listed (`stable_ginv.viz.rst`, `stable_ginv.viz.panels.rst`, …). The `docs/sphinx/api/` files are git-ignored — do not stage them.

- [ ] **Step 5: Harden CI to fail on future doc warnings**

In `.github/workflows/docs.yml`, change the build line:
```yaml
        run: python -m sphinx -b html docs/sphinx docs/sphinx/_build/html
```
to:
```yaml
        run: python -m sphinx -W --keep-going -b html docs/sphinx docs/sphinx/_build/html
```

- [ ] **Step 6: Verify nothing else regressed**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -m pyflakes stable_ginv/viz/plot_masking_sweep_csv.py
conda run -n stable-ginv python -m pytest tests/ -q | tail -1   # 117 passed
git diff --check
```

- [ ] **Step 7: Commit (two commits — separate categories)**

```bash
git add docs/sphinx/conf.py .github/workflows/docs.yml
git commit -m "build: mock pandas + define |grad| subst so Sphinx renders viz; -W in CI"
git add stable_ginv/viz/plot_masking_sweep_csv.py
git commit -m "docs: fix reST literal-block indentation in plot_masking_sweep_csv docstring"
```

---

### Task 2: Module-level docstrings (all `__init__.py` and modules missing one)

**Files (modify — add a module docstring as the first statement; do not touch code):**
`stable_ginv/metrics/__init__.py`, `stable_ginv/metrics/grad_match.py`, `stable_ginv/metrics/image_metrics.py`, `stable_ginv/metrics/jacobian.py`, `stable_ginv/masking/__init__.py`, `stable_ginv/masking/_compute.py`, `stable_ginv/masking/_helpers.py`, `stable_ginv/recon/__init__.py`, `stable_ginv/registry/__init__.py`, `stable_ginv/stats/__init__.py`, `stable_ginv/io/__init__.py`, `stable_ginv/experiment/__init__.py`, `stable_ginv/experiment/results.py`, `stable_ginv/jacobian/cli.py`, `stable_ginv/jacobian/sweep.py`.

(`masking/strategies.py` and `masking/facade.py` module docstrings are added in Task 3; `cli/batch.py` in Task 6.)

- [ ] **Step 1: Add each module docstring**

Insert each as the file's first line (before imports). Use exactly these (accurate to each module's verified contents):

- `stable_ginv/metrics/__init__.py`:
  ```python
  """Metrics: image quality (PSNR/SSIM/TV), gradient matching, and Jacobian rank."""
  ```
- `stable_ginv/metrics/grad_match.py`:
  ```python
  """Gradient-matching loss between observed and dummy gradients (cosine / L2 / sim)."""
  ```
- `stable_ginv/metrics/image_metrics.py`:
  ```python
  """Image-quality metrics: PSNR from MSE, batched SSIM, and total variation."""
  ```
- `stable_ginv/metrics/jacobian.py`:
  ```python
  """Jacobian construction and numerical rank of the per-sample gradient map."""
  ```
- `stable_ginv/masking/__init__.py`:
  ```python
  """Gradient masking: strategy classes, the STRATEGY_REGISTRY, and the Masker facade."""
  ```
- `stable_ginv/masking/_compute.py`:
  ```python
  """Low-level mask computation: keep-id and per-entry mask builders by gradient size."""
  ```
- `stable_ginv/masking/_helpers.py`:
  ```python
  """Internal masking helpers: parameter-index lookups and gradient-magnitude scoring."""
  ```
- `stable_ginv/recon/__init__.py`:
  ```python
  """Reconstruction: the optimization runner, early-stop policy, and label inference."""
  ```
- `stable_ginv/registry/__init__.py`:
  ```python
  """Result registries: key hashing, JSON load/save, entry updates, and CSV summaries."""
  ```
- `stable_ginv/stats/__init__.py`:
  ```python
  """Statistics: NaN-safe aggregates and paired comparisons with confidence intervals."""
  ```
- `stable_ginv/io/__init__.py`:
  ```python
  """I/O utilities: storage-path resolution, safe filesystem writes, and CSV/text helpers."""
  ```
- `stable_ginv/experiment/__init__.py`:
  ```python
  """Batch orchestration: the experiment runner, result aggregation, and restart selection."""
  ```
- `stable_ginv/experiment/results.py`:
  ```python
  """Result aggregation, paired-report construction, and experiment-results CSV rows."""
  ```
- `stable_ginv/jacobian/cli.py`:
  ```python
  """CLI for the Jacobian rank-vs-gradient-budget sweep (multiprocessing + plotting)."""
  ```
- `stable_ginv/jacobian/sweep.py`:
  ```python
  """Compute core for the Jacobian rank sweep: per-budget rank over a gradient budget."""
  ```

- [ ] **Step 2: Verify**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python scripts/_docstring_audit.py stable_ginv | grep '<module>' || echo "no modules missing docstrings"
conda run -n stable-ginv python -m pyflakes stable_ginv
conda run -n stable-ginv python -m pytest tests/ -q | tail -1
```
Expected: no `<module>` lines except (acceptably) `cli/batch.py`, `masking/strategies.py`, `masking/facade.py` which are handled in Tasks 3/6; 117 passed; pyflakes clean.

- [ ] **Step 3: Commit**

```bash
git add stable_ginv/
git commit -m "docs: add module-level docstrings across stable_ginv subpackages"
```

---

### Task 3: Masking strategy & facade docstrings

**Files:**
- Modify: `stable_ginv/masking/strategies.py` (module docstring + `MaskStrategy` + every `apply` method on the 14 strategy classes)
- Modify: `stable_ginv/masking/facade.py` (module docstring + `Masker.__init__` + `Masker.apply`)

Behavior is golden-locked by `tests/golden/test_masking_golden.py` — docstrings must describe, not change, the dispatch.

- [ ] **Step 1: Document `strategies.py`**

Add module docstring as line 1:
```python
"""Masking strategies: one class per ``mask_mode`` plus the STRATEGY_REGISTRY map.

Each strategy exposes ``apply(net, original_dy_dx, prefixes, prefix_layer_fracs,
gradsize_topk, gradsize_topfrac, gradsize_metric)`` and returns a
``(keep_ids, entry_masks)`` tuple in which exactly one element is non-None:
whole-parameter selection returns ``keep_ids``; per-entry selection returns
``entry_masks``. ``STRATEGY_REGISTRY`` maps each ``mask_mode`` name to its class.
"""
```

Add a docstring to the `MaskStrategy` Protocol:
```python
class MaskStrategy(Protocol):
    """Protocol for masking strategies: ``apply(...) -> (keep_ids, entry_masks)``."""
```

Add a one-line docstring to each strategy's `apply` method describing its selection. Use exactly (these match the verified bodies):

- `_IdlgStrategy.apply`: `"""Keep all parameters (no masking) — the iDLG baseline."""`
- `_GradsizeTopKStrategy.apply`: `"""Keep the top-k whole parameters ranked by gradient magnitude."""`
- `_GradsizeTopFracStrategy.apply`: `"""Keep the top fraction of whole parameters ranked by gradient magnitude."""`
- `_GradsizeTopKEntriesStrategy.apply`: `"""Keep the top-k individual gradient entries globally."""`
- `_GradsizeTopFracEntriesStrategy.apply`: `"""Keep the top fraction of individual gradient entries globally."""`
- `_GradsizeTopKEntriesLayerStrategy.apply`: `"""Keep top-k entries per layer over all params (no VGG-classifier exclusion)."""`
- `_GradsizeTopFracEntriesLayerStrategy.apply`: `"""Keep the top fraction of entries per layer (VGG classifier excluded)."""`
- `_PrefixTopKStrategy.apply`: `"""Keep top-k whole params within the named prefix groups."""`
- `_PrefixTopFracStrategy.apply`: `"""Keep the top fraction of whole params within the named prefix groups."""`
- `_PrefixTopKEntriesStrategy.apply`: `"""Keep top-k entries among parameters in the named prefix groups."""`
- `_PrefixTopFracEntriesStrategy.apply`: `"""Keep the top fraction of entries among params in the named prefix groups."""`
- `_PrefixTopKEntriesLayerStrategy.apply`: `"""Keep top-k entries per layer within the named prefix groups."""`
- `_PrefixTopFracEntriesLayerStrategy.apply`: `"""Keep the top fraction of entries per layer within the named prefix groups."""`
- `_PrefixStrategy.apply`: `"""Keep all parameters whose names match the given prefixes."""`

- [ ] **Step 2: Document `facade.py`**

Add module docstring as line 1:
```python
"""Masker facade: resolves a (method, mask_mode) pair to a strategy and applies it."""
```
Read `Masker.__init__` and `Masker.apply`, then add NumPy-style docstrings. `__init__` stores the selection inputs; `apply` runs the chosen strategy against a net and its observed gradients, returning `(keep_ids, entry_masks)`. Document the real parameters from the signatures (`method`, `mask_mode`, `prefixes`, `prefix_layer_fracs`, `gradsize_topk`, `gradsize_topfrac`, `gradsize_metric` for `__init__`; the `apply` signature as written).

- [ ] **Step 3: Verify**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python scripts/_docstring_audit.py stable_ginv/masking
conda run -n stable-ginv python -m pytest tests/golden/test_masking_golden.py tests/golden/test_registry_key_golden.py -q | tail -1
conda run -n stable-ginv python -m pyflakes stable_ginv/masking
```
Expected: `OK: all public symbols in stable_ginv/masking documented`; golden tests pass.

- [ ] **Step 4: Commit**

```bash
git add stable_ginv/masking/
git commit -m "docs: document masking strategies, registry contract, and Masker facade"
```

---

### Task 4: Reconstruction & experiment docstrings

**Files:**
- Modify: `stable_ginv/recon/runner.py` (`ReconstructionRunner.__init__`, `.run`, and the nested `closure`)
- Modify: `stable_ginv/experiment/results.py` (`ResultAggregator.__init__`, and any other flagged public method/class — re-check with the auditor)
- Modify: `stable_ginv/experiment/runner.py` (`BatchExperimentRunner.__init__`, `.run`)

Behavior is golden-locked by `tests/golden/test_recon_worker_golden.py`, `test_baseline_summary_csv_golden.py`, and the CSV-row golden. Read each method first.

- [ ] **Step 1: Document `recon/runner.py`**

Read `ReconstructionRunner.__init__` and `.run` and write NumPy-style docstrings describing the optimization loop (owns dummy-data optimization; `.run` performs the reconstruction and returns its results). Give the nested `closure` a one-line docstring, e.g. `"""L-BFGS closure: zero grads, recompute gradient-matching loss, backprop."""` (verify against the actual body before finalizing the wording).

- [ ] **Step 2: Document `experiment/results.py` and `experiment/runner.py`**

Run `python scripts/_docstring_audit.py stable_ginv/experiment` to get the exact remaining symbols. Document `ResultAggregator` / `RestartSelector` (`__init__` and any public methods), and `BatchExperimentRunner.__init__` / `.run` (GPU scheduling, multiprocessing, output ordering, abort-on-worker-failure). Read each before writing.

- [ ] **Step 3: Verify**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python scripts/_docstring_audit.py stable_ginv/recon
conda run -n stable-ginv python scripts/_docstring_audit.py stable_ginv/experiment
conda run -n stable-ginv python -m pytest tests/ -q | tail -1
conda run -n stable-ginv python -m pyflakes stable_ginv/recon stable_ginv/experiment
```
Expected: both auditor runs report OK; 117 passed.

- [ ] **Step 4: Commit**

```bash
git add stable_ginv/recon/ stable_ginv/experiment/
git commit -m "docs: document reconstruction runner and batch-experiment orchestration"
```

---

### Task 5: I/O & Jacobian symbol docstrings

**Files:**
- Modify: `stable_ginv/io/fs.py` (`Tee` class + its `__init__`, `write`, `flush`)
- Modify: `stable_ginv/jacobian/cli.py` (`main`)
- Modify: `stable_ginv/jacobian/sweep.py` (nested `progress_fn`)

(Module docstrings for these files were added in Task 2.)

- [ ] **Step 1: Document `io/fs.py` `Tee`**

Read the `Tee` class. Add a class docstring (a writable stream that fans writes out to multiple underlying streams — used to mirror stdout to a log file) and one-line docstrings on `__init__`, `write`, `flush` describing each. Example shape:
```python
class Tee:
    """File-like object that mirrors writes to several underlying streams."""
    def __init__(self, *streams):
        """Store the target streams to fan out to."""
    def write(self, data):
        """Write ``data`` to every target stream."""
    def flush(self):
        """Flush every target stream."""
```
(Confirm method names/behavior against the actual body before finalizing.)

- [ ] **Step 2: Document `jacobian/cli.py` `main` and `jacobian/sweep.py` `progress_fn`**

Read `main` (parses args, runs the sweep across budgets via multiprocessing, writes the summary CSV + plot) and write a NumPy-style docstring. Give the nested `progress_fn` a one-line docstring describing its role (per-budget progress callback). Verify wording against the bodies.

- [ ] **Step 3: Verify**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python scripts/_docstring_audit.py stable_ginv/io
conda run -n stable-ginv python scripts/_docstring_audit.py stable_ginv/jacobian
conda run -n stable-ginv python -m pytest tests/golden/test_jacobian_summary_golden.py -q | tail -1
conda run -n stable-ginv python -m pyflakes stable_ginv/io stable_ginv/jacobian
```
Expected: both auditor runs OK; jacobian golden passes.

- [ ] **Step 4: Commit**

```bash
git add stable_ginv/io/ stable_ginv/jacobian/
git commit -m "docs: document io.fs.Tee and the Jacobian sweep CLI"
```

---

### Task 6: CLI batch docstrings

**Files:**
- Modify: `stable_ginv/cli/batch.py` (module docstring + `main` + the panel/gif/restart wrappers: `create_panel_buffers`, `append_result_to_panel_buffers`, `flush_recon_panel`, `save_recon_gif`, `save_restart_curve`, `save_restart_images`)

`stable_ginv/cli/batch.py` is the body behind `iDLG_mask.py`. **Do not change argparse setup, flags, defaults, or help strings** — only add docstrings.

- [ ] **Step 1: Document the module and `main`**

Add module docstring as line 1:
```python
"""Batch CLI entry point (behind ``iDLG_mask.py``): parses args, schedules GPU/CPU
workers, runs the masked/baseline experiments, and writes registries, CSVs, and plots.
"""
```
Read `main` and write a NumPy-style docstring describing the end-to-end batch run. Do not alter parser construction.

- [ ] **Step 2: Document the six wrapper functions**

Read each; most delegate to `stable_ginv.viz`. Add one-line docstrings describing each delegation, e.g.:
```python
def create_panel_buffers(...):
    """Create the per-method image buffers backing the reconstruction panel."""
```
Verify each signature/behavior against the body; document real parameters where non-obvious.

- [ ] **Step 3: Verify**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python scripts/_docstring_audit.py stable_ginv/cli
conda run -n stable-ginv python iDLG_mask.py --help | head -5
conda run -n stable-ginv python -m pytest tests/ -q | tail -1
conda run -n stable-ginv python -m pyflakes stable_ginv/cli
```
Expected: auditor OK; `--help` still works and is unchanged; 117 passed.

- [ ] **Step 4: Commit**

```bash
git add stable_ginv/cli/
git commit -m "docs: document the batch CLI entry point and its viz wrappers"
```

---

### Task 7: Visualization docstrings (mind the `--help` constraint)

**Files:**
- Modify: `stable_ginv/viz/plot_combined_masking_sweep_summary.py` (`main`)
- Modify: `stable_ginv/viz/plot_masking_sweep_csv.py` (`main`)
- Modify: `stable_ginv/viz/plot_normality_scatter.py` (`main`)
- Modify: `stable_ginv/viz/plot_paired_masking_violin.py` (`main`)
- Modify: `stable_ginv/viz/plot_model_parameter_counts.py` (`count_parameters`, `count_parameters_without_classifier`, `get_parameter_counts`, `save_parameter_count_plot`, `main`) — **do NOT edit the module docstring** (`description=__doc__`)
- Modify: `stable_ginv/viz/plot_rank_reconstruction.py` (`main`, nested `closure`) — **do NOT edit the module docstring** (`description=__doc__`)

- [ ] **Step 1: Add `main` docstrings to the four standalone plot CLIs**

For each of `plot_combined_masking_sweep_summary.py`, `plot_masking_sweep_csv.py`, `plot_normality_scatter.py`, `plot_paired_masking_violin.py`: read `main`, add a NumPy-style docstring describing what it reads and what figure/CSV it writes. Do not change parser/flags/filenames.

- [ ] **Step 2: Document `plot_model_parameter_counts.py` functions (NOT the module docstring)**

Read each of `count_parameters`, `count_parameters_without_classifier`, `get_parameter_counts`, `save_parameter_count_plot`, `main` and add NumPy-style docstrings (parameters/returns from the real signatures). Leave the module docstring untouched.

- [ ] **Step 3: Document `plot_rank_reconstruction.py` `main` + `closure` (NOT the module docstring)**

Add a NumPy-style docstring to `main` and a one-line docstring to the nested `closure`. Leave the module docstring untouched.

- [ ] **Step 4: Verify `--help` text is byte-identical for the two coupled CLIs**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
# These two reuse __doc__ as --help; confirm wrappers still print help without error.
conda run -n stable-ginv python -m stable_ginv.viz.plot_rank_reconstruction --help | head -3
conda run -n stable-ginv python -m stable_ginv.viz.plot_model_parameter_counts --help | head -3
git diff stable_ginv/viz/plot_rank_reconstruction.py | grep -E '^[+-]' | grep -v '^[+-][+-]' | grep -iE 'def closure|def main|"""' || true
```
Confirm the diff for those two files touches only `main`/`closure` docstrings (and never lines inside the top-of-file `"""..."""` module docstring).

- [ ] **Step 5: Verify the rest**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python scripts/_docstring_audit.py stable_ginv/viz
conda run -n stable-ginv python -m pytest tests/ -q | tail -1
conda run -n stable-ginv python -m pyflakes stable_ginv/viz
```
Expected: auditor OK for `stable_ginv/viz`; 117 passed; pyflakes clean.

- [ ] **Step 6: Commit**

```bash
git add stable_ginv/viz/
git commit -m "docs: document viz plot CLIs (preserving description=__doc__ help text)"
```

---

### Task 8: Final verification, cleanup & handover

**Files:**
- Delete: `scripts/_docstring_audit.py` (throwaway dev tool — never committed)
- Modify: `docs/handover/HANDOVER_RESTRUCTURE.md` (mark Phase 9 done)
- Modify: `docs/handover/README.md` (bump "Last updated")

- [ ] **Step 1: Full package docstring audit (must be empty)**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python scripts/_docstring_audit.py stable_ginv
conda run -n stable-ginv python scripts/_docstring_audit.py stable_ginv/config.py
```
Expected: `OK: all public symbols ... documented` (root scan covers config.py via the `stable_ginv.py` glob fallback; the second call confirms config.py explicitly).

- [ ] **Step 2: Full verification gate**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda run -n stable-ginv python -m pytest tests/ -q | tail -1            # 117 passed
conda run -n stable-ginv python -m pyflakes stable_ginv                   # clean
conda run -n stable-ginv python iDLG_mask.py --help >/dev/null && echo "help OK"
rm -rf docs/sphinx/api /tmp/sphinx_final
conda run -n stable-ginv python -m sphinx -W -b html docs/sphinx /tmp/sphinx_final 2>&1 | tail -2
git diff --check
```
Expected: 117 passed; pyflakes clean; help OK; `build succeeded` with no warnings; `git diff --check` clean.

- [ ] **Step 3: Remove the throwaway auditor and confirm it isn't staged**

```bash
rm -f scripts/_docstring_audit.py
git status --porcelain scripts/_docstring_audit.py   # expect no output
```

- [ ] **Step 4: Update the handover Status**

In `docs/handover/HANDOVER_RESTRUCTURE.md`, change the Phase 9 status line:
```
- [ ] Phase 9 — docstrings + fill Sphinx API pages + polish.
```
to:
```
- [x] Phase 9 — docstrings + Sphinx API pages + polish. NumPy-style docstrings across
      all stable_ginv subpackages; pandas added to autodoc mocks and a `|grad|` rst
      substitution defined so the recursive autosummary renders every module; CI builds
      docs with `-W`. Build is warning-free. All phases complete.
```
Also update the `## Next phase` section: replace its body with a one-line note that the restructure is complete and the remaining deferred moves (listed below it) are optional follow-ups.

- [ ] **Step 5: Bump the handover README date**

In `docs/handover/README.md`, change `**Last updated:** 2026-06-04` to `**Last updated:** 2026-06-06`.

- [ ] **Step 6: Commit**

```bash
git add docs/handover/HANDOVER_RESTRUCTURE.md docs/handover/README.md
git commit -m "docs: mark Phase 9 complete in handover; restructure finished"
```

- [ ] **Step 7: Refresh the graphify knowledge graph (project convention)**

```bash
graphify update . 2>/dev/null || echo "graphify unavailable — skip"
```
(AST-only, no API cost. Do not commit `graphify-out/` unless the repo already tracks it.)

---

## Self-Review (performed at plan-writing time)

- **Spec coverage:** Phase 9 spec items are "docstring pass" (Tasks 2–7), "fill Sphinx API pages (autosummary)" (Task 1 — the recursive autosummary now imports & renders all 11 subpackages incl. viz), and "final cleanup" (Task 8). Section 4's "exhaustive docstring pass" and napoleon NumPy style are honored. ✓
- **Invariant coverage:** The `--help`/`description=__doc__` coupling for `plot_rank_reconstruction.py` and `plot_model_parameter_counts.py` is called out and the `|grad|` fix routed through `conf.py` (not the docstring). No code moves; goldens run after every behavior-adjacent task. ✓
- **Placeholder scan:** All infra edits (conf.py, docs.yml, the masking_sweep_csv docstring) are exact and were build-verified with `-W`. Docstring tasks name every target symbol and the file-scoped auditor is the completion gate; the strategy `apply` docstrings are given verbatim. Function docstrings that require reading parameter detail instruct "read the body first" with a worked example — intentional for a docstring phase, not a vague placeholder. ✓
- **Consistency:** The auditor script name (`scripts/_docstring_audit.py`), the `-W` flag, and the per-subpackage commit scoping are used consistently across tasks; it is created in the verification section and deleted in Task 8. ✓
