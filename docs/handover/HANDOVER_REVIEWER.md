# Handover: Code Reviewer

**Audience:** An agent or developer performing a targeted code review or cleanup task. Read this before touching any file.

**Last updated:** 2026-05-22

---

## 1. Codebase Overview

This is a Python 3.13 research codebase (~4,740 lines at root level, plus ~2,000 lines in `invertinggradients/`) for running gradient inversion attack experiments. It is a **bachelor's thesis project** — correctness of the research logic is more important than code elegance.

**Do not break:**
- Gradient masking logic (this is the core research contribution)
- Reconstruction optimization loop
- Metric computations (PSNR, SSIM, Jacobian rank)
- CSV output format (downstream analysis scripts depend on column names)
- Any logic inside `invertinggradients/` (it is a git submodule, treat as read-only)

---

## 2. File Inventory and Priority

| File | Lines | Review Priority | Notes |
|------|-------|-----------------|-------|
| `Misc_functions.py` | ~1170 | HIGH | Largest file; contains unrelated concerns mixed together |
| `iDLG_mask.py` | ~937 | HIGH | Main entry point; significant code duplication with `run_single_exp.py` |
| `run_single_exp.py` | ~776 | HIGH | Most complete experiment runner; source of truth for algorithm |
| `run_single_exp_batch.py` | ~592 | MEDIUM | Simplified batch variant; partially redundant with above |
| `jacobian_parallel.py` | ~414 | MEDIUM | Parallel Jacobian computation; reviewed for correctness |
| `Network.py` | ~198 | LOW | Clean; small changes only |
| `iDLG_original.py` | ~337 | LOW | Baseline script; may be simplified or left as-is |
| `jacobian_rank_sweep.py` | ~261 | LOW | Sweep script; fine as-is |
| `Dataset.py` | ~46 | LOW | Simple; no issues |
| `consts.py` | ~9 | LOW | Constants file; see duplication issue below |

---

## 3. Specific Issues to Address

### Issue 1: `Misc_functions.py` does too many things (~1170 lines)

This single file contains: gradient masking logic, PSNR/SSIM computation, Jacobian rank, network builder, LR scheduler, GIF/panel saving, CSV I/O, and baseline registry management. These are unrelated concerns.

**Proposed split:**
- `masking.py` — `build_gradient_mask`, `get_keep_ids`, `get_keep_ids_by_gradsize`, `get_entry_masks_by_gradsize`, `get_prefix_keep_ids`, `get_entry_masks_by_prefix_group`, `get_keep_ids_by_prefix_group`, `flatten_observed_gradients`
- `metrics.py` — `compute_psnr_from_mse`, `compute_ssim_batch`, `total_variation`, `compute_jacobian_rank`, `compute_grad_match_loss`
- `visualization.py` — `save_recon_panel`, `save_recon_gif`
- `io_utils.py` — `save_baseline_registry`, `load_baseline_registry`, `paired_summary`
- `training_utils.py` — `build_network`, `make_scheduler`, `weights_init` (or merge into `Network.py`)

**Important:** After splitting, update all imports in `iDLG_mask.py`, `run_single_exp.py`, `run_single_exp_batch.py`, `jacobian_parallel.py`, `jacobian_rank_sweep.py`.

### Issue 2: Duplicate normalization constants

Normalization constants (mean/std per dataset) appear in three places:
- `consts.py` (root) — used by entry point scripts
- `Misc_functions.py` (inline, around line 20–40) — used by `build_network` and helpers
- `invertinggradients/inversefed/consts.py` — used by the submodule

The root `consts.py` and `Misc_functions.py` copies must be kept in sync. **Fix:** Remove the inline copies from `Misc_functions.py` and import from `consts.py` instead. Do not touch `invertinggradients/`.

### Issue 3: Mixed naming conventions

The `config` dict uses inconsistent casing:
- `'Iteration'` (capitalized, should be `'num_iterations'`)
- `'MASK_MODE'`, `'NETWORK_NAME'` (SCREAMING_SNAKE — reasonable for constants but inconsistent)
- `'lr'`, `'channel'`, `'num_dummy'` (lowercase snake_case)

**Fix for Phase 2 only:** Standardize to snake_case throughout the config dict. This is a breaking change across all entry points — do not do this piecemeal or scripts will silently use wrong values.

### Issue 4: Commented-out code blocks

Several commented-out blocks exist across entry points:
- `run_single_exp.py:68–70` — commented loop over `net.named_parameters()`
- `run_single_exp.py:10` — commented `from skimage.metrics import structural_similarity as ssim`
- `iDLG_mask.py` — scattered commented-out debug prints

**Fix:** Remove these. They are noise that makes it harder to read the active logic.

### Issue 5: No type hints

No function signatures have type annotations. This makes it hard to understand what masking functions return (`keep_ids` is a set, `entry_masks` is a list of tensors or None).

**Fix for Phase 2:** Add type hints to all public functions in `Misc_functions.py` first, then propagate.

### Issue 6: `original/` directory

Contains an older version of the code. Nothing imports from it.

**Fix:** Delete it or archive it to a git tag. Confirm with the project owner before deleting.

### Issue 7: Code duplication between entry points

`iDLG_mask.py` and `run_single_exp.py` share large blocks of identical logic (network setup, dataset loading, gradient computation, masking, optimization loop). This means bug fixes need to be applied in multiple places.

**Fix for Phase 2:** Extract the shared experiment logic into a reusable function in a new `experiment.py` module.

---

## 4. What NOT to Change

- **Any gradient masking logic** in `Misc_functions.py` (`build_gradient_mask`, `get_keep_ids*`, `get_entry_masks*`, `get_prefix_*`, `flatten_observed_gradients`) — this is the core research contribution. Changing variable names or logic here can silently break experimental results.
- **`compute_jacobian_rank`** — the rank computation algorithm is mathematically precise. Do not refactor for "cleanliness" without understanding the math.
- **`compute_grad_match_loss`** — the cosine similarity implementation must match the original paper exactly.
- **CSV column names** — downstream analysis and plotting scripts read these by name. Renaming breaks analysis.
- **`iDLG_original.py`** — kept as a baseline comparison. Leave it alone.
- **`invertinggradients/`** — it is a git submodule. Do not commit changes to it from this repo.
- **Dataset normalization constants** in `consts.py` — these are manually validated (LFW was computed in `testing/compute_lfw_stats.py`). Do not "correct" them without rerunning the validation script.

---

## 5. Patterns to Follow

**Config dict pattern:** All experiments are configured via a plain Python dict passed to worker functions. Keep this pattern — do not introduce argparse or a config file parser without discussing with the project owner first.

**Multi-GPU worker pattern:** `iDLG_mask.py` spawns one process per GPU via `torch.multiprocessing.spawn`. Results are collected via a `multiprocessing.Manager().Queue()`. Follow this pattern for any new parallel experiment scripts.

**Reproducibility:** Every worker sets `torch.manual_seed(seed)`, `torch.cuda.manual_seed_all(seed)`, and `np.random.seed(seed)` using `seed = config['run_id'] + idx_net + 1`. Preserve this pattern.

**Matplotlib Agg backend:** `Misc_functions.py` line 1–2 sets `matplotlib.use("Agg")` before importing pyplot. This must remain the first import in any file that uses matplotlib, or it will crash on headless servers.

---

## 6. How to Verify Nothing Is Broken

There is no pytest suite. Verification is manual:

1. **Dataset loading test:**
   ```bash
   python testing/lfw_test.py
   ```
   Should load LFW, run a forward pass, and print gradient norms without errors.

2. **Quick smoke test (CPU):**
   Edit `run_single_exp.py`: set `device = 'cpu'`, `Iteration = 10`, `NUM_RESTARTS = 1`, `COMPUTE_JACOBIAN_RANK = False`. Run:
   ```bash
   python run_single_exp.py
   ```
   Should complete without error and produce a reconstructed image PNG.

3. **Metric sanity check:**
   After a run, verify PSNR values are in the range 15–40 dB for CIFAR-10 with `resnet18`. Values outside this range indicate a bug in normalization or loss computation.

4. **Masking smoke test:**
   Set `MASK_MODE = 'gradsize_topfrac'`, `GRADSIZE_TOPFRAC = 0.5`. Run a short experiment. If PSNR for masked is worse than unmasked, the masking is working. If they are identical, the mask is not being applied.
