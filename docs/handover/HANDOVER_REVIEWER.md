# Handover: Code Reviewer

**Audience:** An agent or developer performing a targeted code review or cleanup task. Read this before touching any file.

**Last updated:** 2026-05-31

---

## 1. Codebase Overview

This is a Python 3.13 research codebase for running gradient inversion attack experiments. It is a **bachelor's thesis project** — correctness of the research logic is more important than code elegance.

**Do not break:**
- Gradient masking logic (this is the core research contribution)
- Reconstruction optimization loop
- Metric computations (PSNR, SSIM, Jacobian rank)
- CSV output format (downstream analysis scripts depend on column names)
- Any logic inside `invertinggradients/` (it is a git submodule, treat as read-only)

**When updating handover files:**
- Keep active handovers current-state only.
- Remove redundant, outdated, and superseded instructions.
- Include enough architecture, command, safety, verification, and pending-work
  context for a new agent to continue without reading historical logs.
- Keep history in Git unless it directly affects existing experiment results.

---

## 2. File Inventory and Priority

**Root entry points**

| File | Review Priority | Notes |
|------|-----------------|-------|
| `iDLG_mask.py` | HIGH | High-level batch runner; CLI orchestration, worker scheduling, registry/CSV write order |
| `run_single_exp.py` | HIGH | Single-experiment runner; source of truth for algorithm |

**`functions/` — core domain logic**

| File | Review Priority | Notes |
|------|-----------------|-------|
| `functions/masking.py` | HIGH | Gradient masking; core research contribution — do not refactor logic |
| `functions/idlg_cli.py` | MEDIUM | `iDLG_mask.py` argparse flags/defaults and validation |
| `functions/experiment_results.py` | MEDIUM | Batch aggregation, restart bests, paired summaries, CSV row builders |
| `functions/io_utils.py` | MEDIUM | Baseline/masked registries, stats helpers, CSV helpers, storage paths |
| `functions/Dataset.py` | LOW | Simple dataset loaders |
| `functions/consts.py` | LOW | All normalization constants (single source of truth) |
| `functions/jacobian_rank_sweep.py` | LOW | Sweep script; fine as-is |

**`helper/` — shared utilities**

| File | Review Priority | Notes |
|------|-----------------|-------|
| `helper/Network.py` | LOW | Model definitions + factory; clean |
| `helper/metrics.py` | MEDIUM | PSNR, SSIM, Jacobian rank, grad match loss |
| `helper/training_utils.py` | LOW | `make_scheduler` |
| `helper/visualization.py` | LOW | Panel PNG, animated GIF, restart curve/image output |
| `helper/plot_masking_sweep_csv.py` | LOW | Reads sweep CSV rows containing `masked_key`, resolves MSE lists from `masked_registry.json`, and writes threshold plots; default `--threshold_mse 0.01` |
| `helper/plots.py` | LOW | Standalone Seaborn forest plots for layer-ablation PSNR confidence intervals |

**`archive/`** — retired scripts (`iDLG_original.py`, `run_single_exp_batch.py`, old visualize/testing scripts). Nothing imports from these.

**`scripts/`** — DTU HPC job scripts only.

---

## 3. Deferred Cleanup Areas

### Mixed naming conventions

The `config` dict uses inconsistent casing:
- `'Iteration'` (capitalized, should be `'num_iterations'`)
- `'MASK_MODE'`, `'NETWORK_NAME'` (SCREAMING_SNAKE — reasonable for constants but inconsistent)
- `'lr'`, `'channel'`, `'num_dummy'` (lowercase snake_case)

**Fix for Phase 2 only:** Standardize to snake_case throughout the config dict. This is a breaking change across all entry points — do not do this piecemeal or scripts will silently use wrong values.

### Limited type hints

No function signatures have type annotations. This makes it hard to understand what masking functions return (`keep_ids` is a set, `entry_masks` is a list of tensors or None).

**Fix for Phase 2 (P2.3):** Add type hints to all public functions in `functions/masking.py`, `helper/metrics.py`, `helper/Network.py`, `functions/Dataset.py`.

### Remaining duplication between entry points

`iDLG_mask.py` has been reduced to high-level orchestration, with CLI parsing in `functions/idlg_cli.py` and repeated aggregation/CSV/restart helpers in `functions/experiment_results.py`. `run_single_exp.py` still owns reconstruction behavior and should stay unchanged unless the algorithm itself is being changed.

**Fix for Phase 2:** If a larger split is needed, extract shared experiment construction into a reusable module without changing reconstruction math, masking behavior, optimizer behavior, CLI flags, CSV columns, or registry keys.

---

## 4. What NOT to Change

- **Any gradient masking logic** in `functions/masking.py` (`build_gradient_mask`, `get_keep_ids*`, `get_entry_masks*`, `get_prefix_*`, `flatten_observed_gradients`) — this is the core research contribution. Changing variable names or logic here can silently break experimental results.
- **`compute_jacobian_rank`** in `helper/metrics.py` — the rank computation algorithm is mathematically precise. Do not refactor for "cleanliness" without understanding the math.
- **`compute_grad_match_loss`** in `helper/metrics.py` — the cosine similarity implementation must match the original paper exactly.
- **CSV column names** — downstream analysis and plotting scripts read these by name. Renaming breaks analysis.
- **Registry key links** — `exp_results_<network>.csv` now has `registry_key` as the last column. iDLG rows point to the baseline registry; masked rows point to the masked registry.
- **`iDLG_original.py`** — kept as a baseline comparison. Leave it alone.
- **`invertinggradients/`** — it is a git submodule. Do not commit changes to it from this repo.
- **Dataset normalization constants** in `consts.py` — these are manually validated (LFW was computed in `archive/testing/compute_lfw_stats.py`). Do not "correct" them without rerunning the validation script.

---

## 5. Patterns to Follow

**Config dict pattern:** `iDLG_mask.py` uses argparse CLI; `run_single_exp.py` uses a plain Python `config` dict passed to worker functions. Keep both patterns as-is — do not convert `run_single_exp.py` to argparse without discussing with the project owner.

**Multi-GPU worker pattern:** `iDLG_mask.py` launches `torch.multiprocessing.Process` workers and collects results through a queue. It supports both normal per-GPU experiment splits and parallel restart scheduling when `num_exp < num_gpus` and `NUM_RESTARTS > 1`.

**Reproducibility:** Every worker sets `torch.manual_seed(seed)`, `torch.cuda.manual_seed_all(seed)`, and `np.random.seed(seed)` using `seed = config['run_id'] + idx_net + 1`. Preserve this pattern.

**Plotting:** Active statistical charts use seaborn (`sns.lineplot`, `sns.barplot`). Matplotlib remains for the Agg backend, figure/axis creation, `imshow` image panels/GIF frames, layout, labels, and saving. Do not convert reconstruction image grids to seaborn heatmaps.

**Matplotlib Agg backend:** `helper/visualization.py` line 2 sets `matplotlib.use("Agg")` before importing pyplot. This must remain the first import in any file that uses matplotlib, or it will crash on headless servers.

**HPC compiled libraries:** DTU HPC jobs need `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"` after conda activation and before Python. Otherwise SciPy/Matplotlib may load `/lib64/libstdc++.so.6` and fail with `CXXABI_1.3.15 not found`.

**Degraded mode:** `iDLG_mask.py::_load_visualization_helpers()` deliberately falls back to no-op plotting helpers if Matplotlib/Seaborn cannot import. `functions/io_utils.py::_load_scipy_stats()` similarly defers SciPy import and skips p-values/Shapiro checks when unavailable. Preserve registry/CSV writes when changing these paths.

---

## 6. How to Verify Nothing Is Broken

**Automated tests (fast — run these first):**
```bash
python -m pytest tests/ -v
```
Tests cover masking routing paths, the FC/label-inference separation (FC is not force-included; prefix lists act as a pure whitelist), per-layer entry modes, gradient flattening, and safe I/O helpers. Should pass quickly on CPU.

**DTU HPC import check:**
```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
python -c "import scipy; import matplotlib; import seaborn; print('imports ok')"
```

**Manual checks:**

1. **Dataset loading test:**
   ```bash
   python archive/testing/lfw_test.py
   ```
   Should load LFW, run a forward pass, and print gradient norms without errors.

2. **Quick smoke test (CPU):**
   Run with `--network resnet18 --dataset cifar10 --num_exp 1 --iteration 10 --num_restarts 1`. Should complete without error and produce output PNG + CSV.

3. **Metric sanity check:**
   After a run, verify PSNR values are in the range 15–40 dB for CIFAR-10 with `resnet18`. Values outside this range indicate a bug in normalization or loss computation.

4. **Masking smoke test:**
   Run `--methods both --mask_mode gradsize_topfrac --gradsize_topfrac 0.5`. If masked PSNR is worse than iDLG, the masking is working. If they are identical, the mask is not being applied.
