# Handover: Code Reviewer

**Audience:** An agent or developer performing a targeted code review or cleanup task. Read this before touching any file.

**Last updated:** 2026-05-29

---

## 1. Codebase Overview

This is a Python 3.13 research codebase for running gradient inversion attack experiments. It is a **bachelor's thesis project** — correctness of the research logic is more important than code elegance.

**Do not break:**
- Gradient masking logic (this is the core research contribution)
- Reconstruction optimization loop
- Metric computations (PSNR, SSIM, Jacobian rank)
- CSV output format (downstream analysis scripts depend on column names)
- Any logic inside `invertinggradients/` (it is a git submodule, treat as read-only)

---

## 2. File Inventory and Priority

**Root entry points**

| File | Review Priority | Notes |
|------|-----------------|-------|
| `iDLG_mask.py` | HIGH | Main batch runner; argparse CLI; saves baseline + masked registries |
| `run_single_exp.py` | HIGH | Single-experiment runner; source of truth for algorithm |

**`functions/` — core domain logic**

| File | Review Priority | Notes |
|------|-----------------|-------|
| `functions/masking.py` | HIGH | Gradient masking; core research contribution — do not refactor logic |
| `functions/io_utils.py` | MEDIUM | Baseline registry, CSV helpers |
| `functions/Dataset.py` | LOW | Simple dataset loaders |
| `functions/consts.py` | LOW | All normalization constants (single source of truth) |
| `functions/jacobian_rank_sweep.py` | LOW | Sweep script; fine as-is |

**`helper/` — shared utilities**

| File | Review Priority | Notes |
|------|-----------------|-------|
| `helper/Network.py` | LOW | Model definitions + factory; clean |
| `helper/metrics.py` | MEDIUM | PSNR, SSIM, Jacobian rank, grad match loss |
| `helper/training_utils.py` | LOW | `make_scheduler` only (`build_network` deleted) |
| `helper/visualization.py` | LOW | Panel PNG and animated GIF output |

**`archive/`** — retired scripts (`iDLG_original.py`, `jacobian_parallel.py`, `run_single_exp_batch.py`, old visualize/testing scripts). Nothing imports from these.

**`scripts/`**

| File | Review Priority | Notes |
|------|-----------------|-------|
| `scripts/plot_masking_sweep_csv.py` | LOW | Reads sweep CSV rows containing `masked_key`, resolves MSE lists from `masked_registry.json`, and writes threshold plots |

---

## 3. Specific Issues to Address

### Issue 1: ~~`Misc_functions.py` split~~ ✅ DONE

Split into `functions/masking.py`, `functions/io_utils.py`, `helper/metrics.py`, `helper/training_utils.py`, `helper/visualization.py`. Original file deleted.

### Issue 2: ~~Duplicate normalization constants~~ ✅ DONE

All dataset stats now live exclusively in `functions/consts.py`. `invertinggradients/inversefed/consts.py` still has its own copy — treat as read-only.

### Issue 3: Mixed naming conventions

The `config` dict uses inconsistent casing:
- `'Iteration'` (capitalized, should be `'num_iterations'`)
- `'MASK_MODE'`, `'NETWORK_NAME'` (SCREAMING_SNAKE — reasonable for constants but inconsistent)
- `'lr'`, `'channel'`, `'num_dummy'` (lowercase snake_case)

**Fix for Phase 2 only:** Standardize to snake_case throughout the config dict. This is a breaking change across all entry points — do not do this piecemeal or scripts will silently use wrong values.

### Issue 4: ~~Commented-out code blocks~~ ✅ DONE

Removed across all entry points — `run_single_exp.py` and `iDLG_mask.py`.

### Issue 5: No type hints

No function signatures have type annotations. This makes it hard to understand what masking functions return (`keep_ids` is a set, `entry_masks` is a list of tensors or None).

**Fix for Phase 2 (P2.3):** Add type hints to all public functions in `functions/masking.py`, `helper/metrics.py`, `helper/Network.py`, `functions/Dataset.py`.

### Issue 6: ~~`original/` directory~~ ✅ DONE

Moved to `archive/`. Nothing imports from it.

### Issue 7: Code duplication between entry points

`iDLG_mask.py` and `run_single_exp.py` share large blocks of identical logic (network setup, dataset loading, gradient computation, masking, optimization loop). This means bug fixes need to be applied in multiple places.

**Fix for Phase 2:** Extract the shared experiment logic into a reusable function in a new `experiment.py` module.

---

## 4. What NOT to Change

- **Any gradient masking logic** in `functions/masking.py` (`build_gradient_mask`, `get_keep_ids*`, `get_entry_masks*`, `get_prefix_*`, `flatten_observed_gradients`) — this is the core research contribution. Changing variable names or logic here can silently break experimental results.
- **`compute_jacobian_rank`** in `helper/metrics.py` — the rank computation algorithm is mathematically precise. Do not refactor for "cleanliness" without understanding the math.
- **`compute_grad_match_loss`** in `helper/metrics.py` — the cosine similarity implementation must match the original paper exactly.
- **CSV column names** — downstream analysis and plotting scripts read these by name. Renaming breaks analysis.
- **Registry key links** — `exp_results_<network>.csv` now has `registry_key` as the last column. iDLG rows point to the baseline registry; masked rows point to the masked registry.
- **`iDLG_original.py`** — kept as a baseline comparison. Leave it alone.
- **`invertinggradients/`** — it is a git submodule. Do not commit changes to it from this repo.
- **Dataset normalization constants** in `consts.py` — these are manually validated (LFW was computed in `testing/compute_lfw_stats.py`). Do not "correct" them without rerunning the validation script.

---

## 5. Patterns to Follow

**Config dict pattern:** `iDLG_mask.py` uses argparse CLI; `run_single_exp.py` uses a plain Python `config` dict passed to worker functions. Keep both patterns as-is — do not convert `run_single_exp.py` to argparse without discussing with the project owner.

**Multi-GPU worker pattern:** `iDLG_mask.py` spawns one process per GPU via `torch.multiprocessing.spawn`. Results are collected via a `multiprocessing.Manager().Queue()`. Follow this pattern for any new parallel experiment scripts.

**Reproducibility:** Every worker sets `torch.manual_seed(seed)`, `torch.cuda.manual_seed_all(seed)`, and `np.random.seed(seed)` using `seed = config['run_id'] + idx_net + 1`. Preserve this pattern.

**Matplotlib Agg backend:** `helper/visualization.py` line 2 sets `matplotlib.use("Agg")` before importing pyplot. This must remain the first import in any file that uses matplotlib, or it will crash on headless servers.

---

## 6. How to Verify Nothing Is Broken

**Automated tests (fast — run these first):**
```bash
python -m pytest tests/ -v
```
44 tests covering masking routing paths, last-FC invariant, per-layer entry modes, gradient flattening, and safe I/O helpers. Should pass quickly on CPU.

**Manual checks:**

1. **Dataset loading test:**
   ```bash
   python testing/lfw_test.py
   ```
   Should load LFW, run a forward pass, and print gradient norms without errors.

2. **Quick smoke test (CPU):**
   Run with `--network resnet18 --dataset cifar10 --num_exp 1 --iteration 10 --num_restarts 1`. Should complete without error and produce output PNG + CSV.

3. **Metric sanity check:**
   After a run, verify PSNR values are in the range 15–40 dB for CIFAR-10 with `resnet18`. Values outside this range indicate a bug in normalization or loss computation.

4. **Masking smoke test:**
   Run `--methods both --mask_mode gradsize_topfrac --gradsize_topfrac 0.5`. If masked PSNR is worse than iDLG, the masking is working. If they are identical, the mask is not being applied.
