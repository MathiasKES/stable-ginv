# Handover: Developer / Researcher

**Audience:** A developer or researcher who is continuing active work on this codebase — adding new masking modes, running new experiments, debugging, or extending the pipeline.

**Last updated:** 2026-05-29

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
├── iDLG_mask.py             Main batch experiment runner (multi-GPU, all masking modes)
├── run_single_exp.py        Detailed single-experiment runner (GIF, Jacobian rank, SSIM)
│
├── functions/               Core domain logic
│   ├── masking.py           ALL gradient masking: build_gradient_mask, get_keep_ids*,
│   │                        get_entry_masks*, flatten_observed_gradients
│   ├── io_utils.py          Baseline registry, paired stats, CSV helpers, parse_prefixes_with_fracs
│   ├── Dataset.py           load_dataset() (shared), MNIST/CIFAR/LFW loaders
│   ├── consts.py            Normalization constants (single source of truth)
│   └── jacobian_rank_sweep.py  Sweep masking params and compute Jacobian rank per config
│
├── helper/                  Shared utilities
│   ├── Network.py           Model factory (get_model handles ALL architectures) + custom CNNs
│   ├── metrics.py           PSNR, SSIM, total variation, Jacobian rank, grad match loss
│   ├── training_utils.py    make_scheduler only (build_network absorbed into get_model)
│   └── visualization.py     save_recon_panel, save_recon_gif
│
├── archive/                 Retired scripts (not imported anywhere):
│                            iDLG_original.py, jacobian_parallel.py,
│                            run_single_exp_batch.py, old visualize/testing scripts
│
├── testing/                 Dataset validation scripts (not pytest — manual runs)
├── scripts/                 HPC job scripts (DTU cluster)
│   └── plot_masking_sweep_csv.py  Plot threshold sweep curves from registry-backed CSV rows
│
└── invertinggradients/      Git submodule — Geiping et al. reference implementation
    └── inversefed/          GradientReconstructor, metrics, models, data loaders
```

---

## 3. Entry Points

### `iDLG_mask.py` — use for large batch experiments
Spawns one process per GPU via `torch.multiprocessing`. Each process runs `N` experiments independently and puts results into a shared queue. Configured by editing the `config` dict near the bottom of the file.

```bash
python iDLG_mask.py
```

Output: PNG reconstructions + CSV table in the path set by `SAVE_PATH`.

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
| `gradsize_topfrac_entries_layer` | Per-tensor independent top-fraction entries — each tensor keeps its own top fraction |
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
- **Matplotlib backend:** `helper/visualization.py` line 2 forces `matplotlib.use("Agg")` for headless operation. Do not call `matplotlib.pyplot` before this runs in any file that uses visualization.
- **HPC scripts:** `scripts/` targets the DTU HPC cluster (LSF job scheduler). The `init.sh` sets up the conda environment from `environment.yml`.
- **No argparse in `run_single_exp.py`:** Configuration is via editing the `config` dict. `iDLG_mask.py` does have full argparse CLI support.
- **Last FC layer always unmasked:** `build_gradient_mask` unconditionally preserves the last `nn.Linear` layer regardless of mask mode. This is intentional — it guarantees the iDLG label-recovery trick always has the gradient it needs. Do not bypass this by calling the individual masking functions directly.
- **LFW normalization constants:** Computed manually in `testing/compute_lfw_stats.py` and hardcoded in `functions/consts.py`. If you change the LFW preprocessing (resize, crop), recompute these.
- **Masking sweep workflow:** The old `--mse_visualise`, `--threshold_mse`, and `--sweep_step` experiment path was removed. Run normal `iDLG_mask.py` commands for each `--gradsize_topfrac`; then call `scripts/plot_masking_sweep_csv.py <sweep_csv> --threshold_mse <value>`. The plot script includes the matching iDLG baseline as the 0% masked point when `results/baselines/idlg_baselines_registry.json` contains the corresponding baseline key.
- **Jacobian dtype comparisons:** `functions/jacobian_rank_sweep.py --both_dtypes` executes the same sweep for `float32` and `float64`, writes one CSV with a `dtype` column, and saves one overlaid plot. If `--qr_pivot` is also enabled, the QR-pivot lines are dashed and use the same color as the corresponding dtype.

---

## 8. Recent Changes (as of 2026-05-29)

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
- `scripts/plot_masking_sweep_csv.py` added. It reads the sweep CSV, resolves each `masked_key` in `results/baselines/masked_registry.json`, computes reconstructed counts from `best_mse_list` using the supplied threshold, and writes `masking_sweep_summary.csv`, `sweep_plot.png`, and `sweep_bar.png`. It also derives the matching baseline key from the masked registry args and includes the baseline as 0% masked when available.
- Main `exp_results_<network>.csv` rows now include `registry_key` as the last column. iDLG rows use the baseline key; masked rows use the masked registry key.

**Session 2026-05-29 (Jacobian rank dtype comparison):**
- `functions/jacobian_rank_sweep.py` — `--dtype {float32,float64}` added. Default is `float64`, matching previous behaviour.
- `functions/jacobian_rank_sweep.py` — `--both_dtypes` added. Runs the same rank sweep for float32 and float64, reseeding before each dtype so randomly initialized models are comparable.
- `functions/jacobian_rank_sweep.py` — output CSV now includes `dtype`; output PNG overlays dtype-specific rank curves in one graph. With `--qr_pivot`, QR curves are dashed in the same color as their dtype's main curve.

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
