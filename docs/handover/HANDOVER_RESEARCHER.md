# Handover: Developer / Researcher

**Audience:** A developer or researcher who is continuing active work on this codebase — adding new masking modes, running new experiments, debugging, or extending the pipeline.

**Last updated:** 2026-05-31

---

## 1. What This Project Does

`stable-ginv` implements and extends the **iDLG gradient inversion attack** (Zhao et al., 2020) for studying privacy in federated learning. The central question is: *how much does gradient masking degrade an adversary's ability to reconstruct private training images from shared gradients?*

The pipeline is:
1. Train or load a neural network.
2. Sample a private image from a dataset and compute its gradients.
3. Apply a gradient mask (zero out some tensors/entries).
4. Run an optimization loop to reconstruct the image from the masked gradients.
5. Measure reconstruction quality (PSNR, SSIM) and gradient information content (Jacobian rank).

The `invertinggradients/` subfolder is a **git submodule** cloned from Geiping et al.'s original repo. It provides `GradientReconstructor` but the main codebase does not use it directly for experiments — it was kept for reference and the `iDLG_original.py` baseline.

---

## 2. Module Map

```
stable-ginv/
├── iDLG_mask.py             High-level batch experiment runner (CLI, scheduling, orchestration)
├── run_single_exp.py        Detailed single-experiment runner (GIF, Jacobian rank, SSIM)
│
├── functions/               Core domain logic
│   ├── masking.py           ALL gradient masking: build_gradient_mask, get_keep_ids*,
│   │                        get_entry_masks*, flatten_observed_gradients
│   ├── idlg_cli.py          argparse definition and validation for iDLG_mask.py
│   ├── experiment_results.py  Batch result aggregation, restart bests, CSV row builders
│   ├── io_utils.py          Baseline/masked registries, stats, CSV helpers, storage paths
│   ├── Dataset.py           load_dataset() (shared), MNIST/CIFAR/LFW loaders
│   ├── consts.py            Normalization constants (single source of truth)
│   └── jacobian_rank_sweep.py  Sweep masking params and compute Jacobian rank per config
│
├── helper/                  Shared utilities
│   ├── Network.py           Model factory (get_model handles ALL architectures) + custom CNNs
│   ├── metrics.py           PSNR, SSIM, total variation, Jacobian rank, grad match loss
│   ├── training_utils.py    make_scheduler only (build_network absorbed into get_model)
│   ├── visualization.py     Reconstruction panels/GIFs plus restart curve/image outputs
│   └── plot_masking_sweep_csv.py  Plot registry-backed masking sweep CSVs
│
├── archive/                 Retired scripts (not imported anywhere):
│                            iDLG_original.py, jacobian_parallel.py,
│                            run_single_exp_batch.py, old visualize/testing scripts
│
├── testing/                 Dataset validation scripts (not pytest — manual runs)
├── scripts/                 HPC job scripts (DTU cluster)
│
└── invertinggradients/      Git submodule — Geiping et al. reference implementation
    └── inversefed/          GradientReconstructor, metrics, models, data loaders
```

---

## 3. Entry Points

### `iDLG_mask.py` — use for large batch experiments
Parses CLI arguments via `functions/idlg_cli.py`, builds worker configs, schedules normal or parallel-restart workers, aggregates results through `functions/experiment_results.py`, and writes CSV/registry outputs. The reconstruction behavior still lives in `run_single_exp.py`.

```bash
python iDLG_mask.py --network resnet18 --dataset cifar10 --num_exp 10 --methods both
```

Output: PNG reconstructions, restart plots/images when enabled, registry JSON files, and CSV tables under the resolved project/HPC results path.

For `--mask_mode gradsize_topfrac_entries_layer`, normal masked runs also append a compact sweep row to `results/masking_sweeps/`. That row stores the command, `topfrac`, and the `masked_key`; per-sample MSE values stay in `results/baselines/masked_registry.json`.

### `run_single_exp.py` — use for deep single-experiment inspection
Same masking logic, but also supports:
- `COMPUTE_JACOBIAN_RANK = True` — compute Jacobian rank before/after masking
- `SAVE_GIF = True` — save reconstruction as animated GIF
- Per-image PSNR and SSIM (not just averages)
- Multiple `METHODS` in one run (`'both'`, `'masked'`, `'unmasked'`)

```bash
python run_single_exp.py
```

### `functions/jacobian_rank_sweep.py` — use for parameter sweeps of rank
Iterates over a grid of masking parameters and logs rank results. Default numerical dtype is `float64`; pass `--dtype float32` to run in single precision, or `--both_dtypes` to run float32 and float64 back-to-back and plot both rank curves in one graph.

**Archived (do not use):** `run_single_exp_batch.py`, `jacobian_parallel.py`, `iDLG_original.py` — moved to `archive/`. Use `iDLG_mask.py` and `run_single_exp.py` instead.

---

## 4. Configuration Reference

All entry points use a `config` dict. Edit it directly in the script — there is no CLI config file. Key fields:

| Key | Type | Meaning |
|-----|------|---------|
| `NETWORK_NAME` | str | `'resnet18'`, `'resnet50'`, `'vgg11'`, `'LeNet'`, `'MediumCNN'`, etc. |
| `NETWORK_TRAINED` | bool | `True` = load ImageNet pretrained weights; `False` = random init |
| `channel` | int | 1 (MNIST), 3 (CIFAR/LFW) |
| `num_classes` | int | 10, 100, 5749 (LFW) |
| `shape_img` | tuple | `(32, 32)` for CIFAR/LFW, `(28, 28)` for MNIST |
| `MASK_MODE` | str | See masking modes below |
| `GRADSIZE_TOPK` | int | Keep this many tensors (for `gradsize_topk`) |
| `GRADSIZE_TOPFRAC` | float | 0.0–1.0, fraction to keep (for `gradsize_topfrac*` modes) |
| `GRADSIZE_METRIC` | str | `'l2'`, `'mean_abs'`, `'sum_abs'` |
| `PREFIXES` | tuple of str | Layer name prefixes for prefix-based modes, e.g. `('conv1', 'layer1')` |
| `PREFIX_LAYER_FRACS` | dict | Per-prefix fractions for `prefix_topfrac_entries_layer` |
| `OPTIMIZER` | str | `'lbfgs'`, `'adam'`, `'adamw'` |
| `lr` | float | Learning rate for reconstruction optimizer |
| `GAMMA` | float | LR decay factor per step (exponential scheduler) |
| `Iteration` | int | Total reconstruction iterations |
| `MAX_ITERATION` | int | L-BFGS inner iterations per step |
| `NUM_RESTARTS` | int | How many random restarts to try; best PSNR is kept |
| `TV_WEIGHT` | float | Total variation regularization weight (0.0 = off) |
| `GRAD_LOSS` | str | `'cos'` (cosine similarity) or `'l2'` |
| `COMPUTE_JACOBIAN_RANK` | bool | Compute Jacobian rank before/after masking |
| `JACOBIAN_MAX_ENTRIES` | int | Cap on gradient entries used for Jacobian (reduces OOM) |
| `num_dummy` | int | Batch size to reconstruct (usually 1) |

**Important:** When `OPTIMIZER = 'lbfgs'`, the code automatically overrides `lr`, `Iteration`, and `MAX_ITERATION` to match the original iDLG paper's values (`lr=1`, `Iteration=300`, `MAX_ITERATION=20`). This is intentional — see `run_single_exp.py:57–67` area.

---

## 5. Gradient Masking Modes

All masking is implemented in `functions/masking.py::build_gradient_mask()`. The function returns either `(keep_ids, None)` for tensor-wise masking or `(None, entry_masks)` for element-wise masking.

| Mode | What it does |
|------|-------------|
| `gradsize_topk` | Keep the K tensors (parameter tensors) with the largest gradient magnitude |
| `gradsize_topfrac` | Keep the top fraction of tensors by gradient magnitude |
| `gradsize_topk_entries` | Keep the K individual gradient scalar entries (across all tensors) with largest magnitude |
| `gradsize_topfrac_entries` | Keep the top fraction of scalar entries across all tensors |
| `gradsize_topk_entries_layer` | Per-tensor independent top-K entries — each parameter tensor keeps its own top K |
| `gradsize_topfrac_entries_layer` | Per-tensor independent top-fraction entries — each tensor keeps its own top fraction. For torchvision VGG models, intermediate classifier layers are excluded by default and only the final classifier layer is fully kept for label inference. |
| `prefix` | Keep only tensors whose parameter name starts with one of the given prefixes |
| `prefix_topk` | Within prefix-matched tensors, keep top-K by magnitude |
| `prefix_topk_entries` | Within prefix-matched tensors, keep top-K scalar entries globally |
| `prefix_topfrac_entries_layer` | Within prefix-matched tensors, keep top fraction per-layer independently |
| `none` | No masking — equivalent to unmasked attack |

**Gradient magnitude metrics** (controlled by `GRADSIZE_METRIC`):
- `l2` — Frobenius norm (default, most stable)
- `mean_abs` — mean of absolute values; robust to outliers
- `sum_abs` — L1 norm; similar to `mean_abs` but also scales with tensor size

---

## 6. How Reconstruction Works

For each experiment, the flow is:
1. Load a random image from the dataset.
2. Forward pass through the network; compute `CrossEntropyLoss`.
3. Backprop to get `dL/dθ` for all parameters — this is the "true gradient."
4. Apply the mask (zero out or exclude gradient tensors/entries).
5. Initialize a "dummy image" from random noise (or zeros for L-BFGS).
6. Optimization loop: minimize `gradient_match_loss(dummy_grad, masked_true_grad) + TV_WEIGHT * TV(dummy_image)`.
7. The **iDLG label recovery trick** recovers the true label from the gradient of the final fully-connected layer's bias: `label = argmin(sum(fc_bias_grad, dim=-1))`. This is exact for single-sample batches.
8. Best reconstruction across restarts (by PSNR) is saved.

**Loss functions:**
- `cos` — `1 - cosine_similarity(dummy_grad_flat, true_grad_flat)` — more robust, default
- `l2` — MSE between gradient vectors

---

## 7. Known Quirks and Gotchas

- **`Iteration` vs `num_iterations`:** Config uses uppercase `Iteration` (not `num_iterations`). This is inconsistent with Python convention and easy to mistype.
- **L-BFGS override:** When `OPTIMIZER = 'lbfgs'`, the code silently overrides `lr`, `Iteration`, and `MAX_ITERATION`. Check `run_single_exp.py` around line 57 if your settings seem to be ignored.
- **`OPTIMIZE_NORM_SPACE`:** Set to `True` whenever `NETWORK_TRAINED = True`. This means the dummy image is optimized in the normalized pixel space matching ImageNet stats. Reconstructions are then un-normalized for saving. If you add a new dataset with different normalization, you need to wire its stats into `functions/consts.py` **and** into the saving/display logic.
- **Normalization constants:** `functions/consts.py` is the single source of truth for the main codebase. `invertinggradients/inversefed/consts.py` is a separate copy inside the submodule — do not modify it.
- **`invertinggradients/` is a submodule:** It has its own `.git`. Do not commit files inside it to the main repo. If you need to update it: `cd invertinggradients && git pull`.
- **Plotting convention:** Statistical plots use seaborn (`sns.lineplot`, `sns.barplot`) with matplotlib only for figure creation, labels, saving, and image rendering (`imshow`). Reconstruction panels and GIF frames remain matplotlib image displays because seaborn does not replace RGB image rendering.
- **Matplotlib backend:** `helper/visualization.py` line 2 forces `matplotlib.use("Agg")` for headless operation. Do not call `matplotlib.pyplot` before this runs in any file that uses visualization.
- **HPC C++ runtime path:** After activating the conda environment on DTU HPC, run `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"` before Python. Without this, compiled SciPy/Matplotlib extensions may load the old system `/lib64/libstdc++.so.6` and fail with `CXXABI_1.3.15 not found`. Add the export to job scripts before the Python command.
- **Optional plotting fallback:** `iDLG_mask.py` lazy-loads visualization helpers. If Matplotlib/Seaborn cannot import, experiments continue without PNG panels, GIFs, or restart plots; registry JSON and CSV outputs are still written so plots can be generated later.
- **Optional SciPy stats fallback:** `functions/io_utils.py` lazy-loads `scipy.stats` only when paired comparisons are needed. If SciPy cannot import, paired p-values and Shapiro-Wilk normality checks are skipped and the CI uses a normal-approximation critical value. Fix the HPC library path before producing final statistical results.
- **HPC scripts:** `scripts/` targets the DTU HPC cluster (LSF job scheduler). The `init.sh` sets up the conda environment from `environment.yml`.
- **No argparse in `run_single_exp.py`:** Configuration is via editing the `config` dict. `iDLG_mask.py` does have full argparse CLI support.
- **Last FC layer always unmasked:** `build_gradient_mask` unconditionally preserves the last `nn.Linear` layer regardless of mask mode. This is intentional — it guarantees the iDLG label-recovery trick always has the gradient it needs. Do not bypass this by calling the individual masking functions directly.
- **LFW normalization constants:** Computed manually in `testing/compute_lfw_stats.py` and hardcoded in `functions/consts.py`. If you change the LFW preprocessing (resize, crop), recompute these.
- **Masking sweep workflow:** Run normal `iDLG_mask.py` commands for each `--gradsize_topfrac`; all runs with the same config (same network/dataset/optimizer/lr/etc.) append to the same CSV in `results/masking_sweeps/` regardless of `--run_id` or `--num_exp`. Then call `helper/plot_masking_sweep_csv.py <sweep_csv>`; its default reconstruction threshold is `--threshold_mse 0.01`. The plot script includes the matching iDLG baseline as the 0% masked point when `results/baselines/idlg_baselines_registry.json` contains the corresponding baseline key. Note: the old experiment-side `--mse_visualise`, `--threshold_mse`, and `--sweep_step` flags were removed from `iDLG_mask.py`.
- **Jacobian dtype comparisons:** `functions/jacobian_rank_sweep.py --both_dtypes` executes the same sweep for `float32` and `float64`, writes one CSV with a `dtype` column, and saves one overlaid plot. If `--qr_pivot` is also enabled, the QR-pivot lines are dashed and use the same color as the corresponding dtype.
- **VGG fraction sweeps:** For `gradsize_topfrac_entries_layer` on VGG, `classifier.0.*` and `classifier.3.*` are excluded automatically. `classifier.6.*` remains fully unmasked because the last-FC layer is required for iDLG label recovery. This substantially reduces VGG sweep runtime and memory use.
- **Row normalisation in rank sweep:** By default, rows of J are L2-normalised before `matrix_rank` to prevent a monotonicity bug (rank decreasing with more rows). Pass `--no_normalisation` to skip this — the default relative threshold then reflects actual gradient magnitudes rather than directions. If using `--no_normalisation`, verify rank is non-decreasing across your row counts for a sanity check.
- **SVD diagnostics:** `--print_svd_info` now prints `sigma_max`, the computed `atol`, whether normalisation was applied, and the bottom-10 singular values. Use this to check whether borderline singular values have comfortable margin above the threshold.

---

## 8. Recent Changes (as of 2026-05-30)

**Session 2026-05-30 (iDLG runner cleanup and plotting convention):**
- `iDLG_mask.py` is now mostly orchestration: CLI parsing, worker config construction, process scheduling, registry/CSV write ordering, and summary printing remain there.
- `functions/idlg_cli.py` owns the `iDLG_mask.py` argparse flags and validation. CLI names and defaults were preserved.
- `functions/experiment_results.py` owns repeated result aggregation, restart best selection, paired summaries, CSV row building, masking sweep CSV writes, and final summary printing.
- `functions/io_utils.py` now includes reusable numeric summary helpers, CSV append helpers, and shared project/HPC storage path resolution. `show_img.py` uses the same storage resolver.
- `helper/visualization.py` now owns reconstruction panel buffering plus restart curve and restart image outputs.
- `helper/plot_masking_sweep_csv.py` now owns registry-backed masking sweep plotting. It moved out of `scripts/`, `--threshold_mse` defaults to `0.01`, output filenames include network/dataset/threshold, the summary CSV includes `network`/`dataset`, generated plot titles show both, and a masked `topfrac=1.0` row is kept separately from the unmasked iDLG baseline.
- Statistical chart plotting in active scripts now uses seaborn. Matplotlib remains the backend for image display, axes labels, layout, and file saving.
- Verification after the cleanup: `python -m pytest tests/ -v` passed with 45 tests.

**Session 2026-05-31 (HPC compiled-library resilience):**
- DTU HPC jobs must prepend `$CONDA_PREFIX/lib` to `LD_LIBRARY_PATH` so compiled SciPy/Matplotlib extensions load conda's `libstdc++.so.6`.
- `iDLG_mask.py` now lazy-loads visualization helpers and continues in no-plot mode if plotting libraries cannot import.
- `functions/io_utils.py` now lazy-loads `scipy.stats`; unavailable SciPy no longer prevents experiment startup.
- `safe_savefig()` catches plotting exceptions and logs a warning instead of aborting the run.

**First session (2026-05-22 / early 2026-05-23):**
- `functions/masking.py` — last FC layer always preserved in `build_gradient_mask` (tensor-wise: union into keep_ids; entry-wise: all-True override); `_grad_magnitude` helper extracted; `gradsize_threshold` mode and parameter removed; `gradsize_topfrac_entries_layer` and `gradsize_topk_entries_layer` modes added (per-tensor independent selection via `get_entry_masks_by_prefix_group` with all param names as groups)
- `functions/Dataset.py` — `load_dataset(dataset, data_path)` added as shared loader used by all entry points; `_Dataset_from_Image` renamed private
- `functions/io_utils.py` — `parse_prefixes_with_fracs(prefixes_str)` added; parses `"conv1:0.5,layer1:1.0,fc"` into `(tuple, dict)`
- `functions/jacobian_rank_sweep.py` — synced with `iDLG_mask.py`: uses `parse_prefixes_with_fracs`, `load_dataset`, correct `pretrained` normalization (imagenet stats when pretrained+RGB), `weights_init` only on custom CNNs, `gradsize_threshold` removed
- `helper/Network.py` — `get_model` now handles all architectures; `pretrained` flag loads ImageNet weights for torchvision models
- `helper/training_utils.py` — `build_network` deleted; only `make_scheduler` remains
- `run_single_exp.py` — 85-line inline masking dispatch removed; uses `build_gradient_mask()` directly; TV now computed on `dummy_data` (normalized space) not de-normalized `x_raw` — matches Geiping et al.
- `tests/test_masking.py` — 24 tests; covers last-FC invariant, per-layer entry modes, all routing paths

**Second session (2026-05-23):**
- `run_single_exp.py` — TV regularization fix: now computes `total_variation(dummy_data)` instead of `total_variation(x_raw)`, matching the Geiping et al. paper. Use `--tv_weight 0.01` for trained-network single-image experiments (paper value).
- `functions/jacobian_rank_sweep.py` — merged serial and parallel (mp.spawn) into one script; `--num_workers 1` (default, serial) to `--num_workers 4`; per-sample grad_norm/nan/inf diagnostics added; rank printed at max row count
- `helper/metrics.py` — Jacobian rank monotonicity fix: rows of J are now normalized before SVD (bounds σ_max ≤ √M); uses fixed `atol = max(rank_tol, max(M,N)·ε·√M)` with `rtol=0.0` — rank no longer declines as row count grows; dead `normalize_rows` parameter removed
- `functions/io_utils.py` — Shapiro-Wilk normality test integrated into `paired_t_ci` (returns `shapiro_stat`, `shapiro_p`) and `paired_summary` (returns `normality_str`); masked registry functions added: `masked_key_from_args`, `load_masked_registry`, `save_masked_registry`, `update_masked_registry`
- `iDLG_mask.py` — saves masked registry to `results/baselines/masked_registry.json` after each masked run; CSV now includes `psnr_normality` and `mse_normality` columns (Shapiro-Wilk result strings for paired PSNR/MSE differences)

**Later sessions (2026-05-24 to 2026-05-27):**
- `iDLG_mask.py` — restart curve CSV/PNG and restart image figure are saved when `NUM_RESTARTS > 1`; parallel restarts are automatic when `num_exp < num_gpus` and `NUM_RESTARTS > 1`.
- `helper/metrics.py` / `functions/jacobian_rank_sweep.py` — Jacobian rank now uses PyTorch's default matrix-rank tolerance after row normalization; prior results with `rank_tol=1e-6` may undercount rank.

**Session 2026-05-29 (masking sweep simplification):**
- `helper/masking_sweep.py` deleted. There is no separate experiment-side sweep runner.
- `iDLG_mask.py` normal runs now handle sweep data for `gradsize_topfrac_entries_layer`: each run still saves a masked registry entry, and appends `command`, `topfrac`, and `masked_key` to a config-specific CSV in `results/masking_sweeps/`. The sweep CSV filename hashes the masked registry comparable args with `gradsize_topfrac` removed, so all fractions for the same setup land in one file.
- `helper/plot_masking_sweep_csv.py` added. It reads the sweep CSV, resolves each `masked_key` in `results/baselines/masked_registry.json`, computes reconstructed counts from `best_mse_list` using the supplied threshold, and writes `masking_sweep_summary_<network>_<dataset>_threshold_<value>.csv`, `sweep_plot_<network>_<dataset>_threshold_<value>.png`, and `sweep_bar_<network>_<dataset>_threshold_<value>.png`. It also derives the matching baseline key from the masked registry args and includes the baseline as 0% masked when available.
- Main `exp_results_<network>.csv` rows now include `registry_key` as the last column. iDLG rows use the baseline key; masked rows use the masked registry key.

**Session 2026-05-29 (Jacobian rank dtype comparison):**
- `functions/jacobian_rank_sweep.py` — `--dtype {float32,float64}` added. Default is `float64`, matching previous behaviour.
- `functions/jacobian_rank_sweep.py` — `--both_dtypes` added. Runs the same rank sweep for float32 and float64, reseeding before each dtype so randomly initialized models are comparable.
- `functions/jacobian_rank_sweep.py` — output CSV now includes `dtype`; output PNG overlays dtype-specific rank curves in one graph. With `--qr_pivot`, QR curves are dashed in the same color as their dtype's main curve.

**Session 2026-05-29 (VGG masking speedup):**
- `functions/masking.py` — For torchvision VGG models with `gradsize_topfrac_entries_layer`, intermediate classifier parameters are excluded from the per-layer mask groups. The existing last-FC override then restores only `classifier.6.weight` and `classifier.6.bias` fully.
- `tests/test_masking.py` — Added a VGG-shaped unit test confirming `classifier.0.*` and `classifier.3.*` are skipped while `classifier.6.*` remains all-True.

**Session 2026-05-30 (masking sweep CSV fix and Jacobian rank CSV improvement):**
- `iDLG_mask.py` — sweep CSV filename hash no longer includes `run_id`, `num_exp`, or `gradsize_topk`. All topfrac values for the same config now append to one CSV regardless of batch size or seed. Existing CSVs on HPC used the old hash; re-plot from `masked_registry.json` if needed.
- `functions/jacobian_rank_sweep.py` — `sample_indices` column added to the rank sweep CSV (semicolon-separated, same order as `per_sample_ranks`). Identify the exact dataset image behind any anomalous rank result without manually reproducing the seed.

**Session 2026-05-29 (Jacobian rank sweep — rank print, normalisation flag, refactor):**
- `functions/jacobian_rank_sweep.py` — Rank is now printed for every row count per sample, not just the maximum. Format: `[i/n] rows=K: rank=R  shape=(K, 3072)`.
- `functions/jacobian_rank_sweep.py` + `helper/metrics.py` — `--no_normalisation` flag added. By default rows of J are L2-normalised before `matrix_rank` (existing behaviour). `--no_normalisation` skips this, making the threshold reflect actual gradient magnitudes. Normalisation can be toggled per run without code changes.
- `helper/metrics.py` — `--print_svd_info` now also prints `sigma_max`, the computed `atol` (`max(M,N)*eps*sigma_max`), and `normalised=True/False` alongside the bottom-10 singular values.
- `functions/jacobian_rank_sweep.py` — Internal refactor: helper functions `_storage_root`, `_save_dir`, `_seed_all`, `_resolve_device`, `_new_rank_results`, `_summarize_rank_results`, `_print_rank_summary`, `_run_serial`, `_run_parallel`, `_run_dtype` extracted from `main()`. No behavioural changes.
- `helper/metrics.py` — `selected_idx.to(device)` moved outside the fwAD column loop (was called 3072× per sample for CIFAR). `_qr_rank` renamed to `_qr_pivot_rows`. In `--independent --qr_pivot` mode, the redundant QR + second SVD is now skipped (rank of a freshly-built J_k cannot change under row reordering).

**Session 2026-05-27 (Jacobian rank sweep improvements — first batch):**
- `functions/jacobian_rank_sweep.py` — Added `per_sample_ranks` column to CSV (semicolon-separated integers, one per sample per row count); per-sample rank list now also printed to stdout after the sweep finishes.
- `functions/jacobian_rank_sweep.py` — `--atol` argument removed entirely; `matrix_rank` now uses PyTorch's own default `rtol` (no user override). Also removed `--gradsize_metric` argument; gradient magnitude is always computed with L2 norm (hardcoded).
- `helper/metrics.py` — `_build_jacobian()` and `_rank_of_J()` extracted as private helpers. `_build_jacobian` auto-selects forward-mode AD (column-wise, `unknowns` passes) when `unknowns < used_entries`, falling back to backward-mode (row-wise) otherwise. Forward-mode is ~10× faster for CIFAR-scale inputs with large `max_entries` (e.g. 3072 unknowns vs 30 000 gradient entries). Requires PyTorch ≥ 2.0 (confirmed 2.8.0 in environment).
- `helper/metrics.py` — `compute_jacobian_rank_sweep()` now builds J once at `max(row_counts)` then slices `J_max[:k]` for each k, eliminating all redundant forward+backward passes.
- `functions/jacobian_rank_sweep.py` — `_print_mask_debug` rewritten: now shows the actual kept and skipped parameter tensor names (and the exact kept/total entry fraction) rather than prefix-based counts, so output is correct for any architecture (not just ResNet).

**Session 2026-05-27 (Jacobian rank sweep improvements — second batch):**
- `functions/jacobian_rank_sweep.py` — `--stepsize` and `--max_row_count` added. Generates row counts as `[stepsize, 2×stepsize, ..., max_row_count]`; overrides `--row_counts` when set. Start is always `stepsize` (not `unknowns`).
- `helper/metrics.py` — `layer_spread` select mode added and subsequently refined (see below).
- `helper/metrics.py` / `functions/jacobian_rank_sweep.py` — `--qr_pivot` added as a standalone boolean flag (not a select mode). After building `J_max` with the chosen `--jacobian_select_mode`, reorders rows via QR decomposition with column pivoting on `J_max^T` — `pivots[i]` is the i-th most linearly independent row. No extra forward/backward passes required (reordering only). Requires `scipy`.
- `helper/metrics.py` / `functions/jacobian_rank_sweep.py` — when `--qr_pivot` is set, `compute_jacobian_rank_sweep` computes ranks for both the select-mode ordering and the QR-pivot ordering in one run. CSV gains three extra columns (`mean_rank_qr`, `std_rank_qr`, `per_sample_ranks_qr`); plot shows both curves; stdout prints two rank tables.

**Session 2026-05-27 (layer_spread rewrite):**
- `helper/metrics.py` — `layer_spread` rewritten to spread the budget across **layer groups** (first component of parameter name: `conv1`, `bn1`, `layer1`, ..., `fc`) rather than individual parameter tensors. Each group gets an equal share of the `max_entries` budget; within each group the top-magnitude entries are selected globally. For ResNet-18 with 7 groups and `max_entries=20000`, each group receives ~2857 entries — enough to capture informative gradient directions from each architectural stage. The original per-tensor approach gave only ~490 entries per tensor which was insufficient for deep conv layers and caused the rank to plateau at ~2631 instead of reaching full rank 3072. Note: `topk_abs` still reaches full rank more efficiently (~7000 entries) because the rank-contributing entries cluster in high-magnitude regions; `layer_spread` is useful when you want guaranteed coverage of every architectural stage regardless of magnitude.

From git log:
- `740209a` — Start row_counts from stepsize instead of unknowns
- `feb5626` — Save both select_mode and qr_pivot ranks separately in sweep output
- `da46fc7` — Make qr_pivot a standalone flag independent of jacobian_select_mode
- `69a9a21` — Add layer_spread and qr_pivot Jacobian select modes
- `a6d88c8` — Add --stepsize and --max_row_count args to Jacobian rank sweep
- `57745ba` — Add forward-mode AD for J columns; fix mask debug display
- `eaa7ffb` — Refactor Jacobian rank sweep: build J once, compute rank for all row counts
- `86a6522` — Print and save per-sample ranks in jacobian rank sweep
- `62f6f47` — Add fixed atol and SVD info to Jacobian rank sweep
- `71f6cd7` — Add masking sweep step arg
- `c5257f3` — Add MSE threshold visualisation script (`helper/visualise_mse_threshold.py`)
- `b071e61` — Delete stale jacobian_parallel files from archive
- `849651c` — Fix review findings: normality key bug, prefix_topfrac routing, fragile FC index, png path, dead code
- `500d89b` — Add normality results to final summary print
- `56c3c3e` — Add mse_normality column to CSV
- `fff0688` — Add masked registry, Shapiro-Wilk, TV fix, and Jacobian rank monotonicity fix
- `35be9ad` — Merge serial + parallel Jacobian rank sweep into one script
- `22c9ad9` — Fix TV normalization (compute on dummy_data not x_raw)
- `426a021` — Sync jacobian_rank_sweep with main experiment scripts, remove gradsize_threshold
- `bd816f8` — Refactor masking internals, enforce last-FC invariant, consolidate model factory

---

## 9. Open Research Questions

- How does Jacobian rank correlate with PSNR/SSIM across different masking modes? (Data exists but analysis is ongoing.)
- Does masking only the final layers (classifier head) adequately protect while preserving model utility?
- How do results scale with larger batch sizes (`num_dummy > 1`)?
- Is there a masking threshold where reconstruction completely fails (phase transition)?
- LFW has 5749 classes — label recovery via the iDLG trick may behave differently. Needs validation.

---

## 10. Environment Setup

```bash
conda env create -f environment.yml
conda activate stable-ginv   # or whatever name is in environment.yml
```

Requires CUDA 12.8. On DTU HPC, load modules first (see `scripts/init.sh`).

For local CPU-only testing, change `device = f'cuda:{device_id}'` to `device = 'cpu'` in the worker function and set `num_gpus = 1`.

---

## 11. Output Interpretation

**PSNR (Peak Signal-to-Noise Ratio):** Higher = better reconstruction. Typical values:
- > 30 dB: visually close to original
- 20–30 dB: recognizable but noisy
- < 20 dB: poor reconstruction / attack fails

**SSIM (Structural Similarity):** 0–1, higher = better. Correlates with perceptual quality better than PSNR. Added recently (commit `45ee353`).

**Jacobian rank:** The rank of `d(gradient_vector) / d(image_pixels)`. Full rank = gradient contains maximum information about the image. Rank drops as masking removes more gradient tensors. Used to quantify *why* masking degrades reconstruction.

**CSV output columns** (masked/both rows): `method`, `timestamp`, `job_id`, `dataset`, `network`, `restarts`, `lr`, `iteration`, `num_exp`, `tv_weight`, `optimizer`, `max_iter`, `history`, `mask_mode`, `prefixes`, `grad_param`, `med_best_loss`, `avg_best_loss`, `med_best_mse`, `avg_best_mse`, `avg_best_psnr`, `std_best_psnr`, `avg_best_ssim`, `std_best_ssim`, `mse_ci`, `mse_significant`, `psnr_ci`, `psnr_significant`, `psnr_normality`, `mse_normality`, `png_path`, `registry_key`

`psnr_normality` / `mse_normality` — Shapiro-Wilk result string on paired differences, e.g. `"normal (W=0.9821, p=0.3412)"` or `"NON-NORMAL (W=0.8123, p=0.0031)"`. Empty for iDLG-only rows.
