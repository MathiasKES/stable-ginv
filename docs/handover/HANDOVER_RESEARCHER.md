# Handover: Developer / Researcher

**Audience:** A developer or researcher who is continuing active work on this codebase — adding new masking modes, running new experiments, debugging, or extending the pipeline.

**Last updated:** 2026-06-02

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
│   ├── training_utils.py    make_scheduler
│   ├── visualization.py     Reconstruction panels/GIFs plus restart curve/image outputs
│   ├── plot_masking_sweep_csv.py  Plot registry-backed masking sweep CSVs
│   └── plots.py             Plot layer-ablation PSNR confidence intervals
│
├── archive/                 Retired scripts (not imported anywhere):
│                            iDLG_original.py, run_single_exp_batch.py,
│                            old visualize/testing scripts
│
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

### `run_single_exp.py` — worker implementation
This module owns reconstruction behavior and is called by `iDLG_mask.py`
workers. It is not a standalone CLI. Use the corresponding `iDLG_mask.py`
flags, such as `--compute_jacobian_rank`, `--save_gif`, and `--methods`.

### `functions/jacobian_rank_sweep.py` — use for parameter sweeps of rank
Iterates over a grid of masking parameters and logs rank results. Default numerical dtype is `float64`; pass `--dtype float32` to run in single precision, or `--both_dtypes` to run float32 and float64 back-to-back and plot both rank curves in one graph.

**Defaults:** independent mode (one J build per row count), no row normalisation, FC in the pool (not forced). Key flags to change defaults:

- `--no_independent` — use pool-slice mode (build J once at max budget, slice for each k). Faster but rank at k depends on pool composition.
- `--normalise` — L2-normalise J rows before `matrix_rank`. Default off: threshold reflects actual gradient magnitudes.
- `--keep_fc` — select from non-FC entries only; FC excluded from Jacobian rows entirely.

**`force_fc` is `False` by default** in `compute_jacobian_rank_sweep` (`helper/metrics.py`). FC entries compete naturally in the pool alongside all other gradient entries.

The scripts under `archive/` are inactive reference material. Use
`iDLG_mask.py` for experiments.

---

## 4. Configuration Reference

`iDLG_mask.py` exposes CLI flags and constructs the worker `config` dict passed
to `run_single_exp.py`. The worker consumes these internal keys:

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
| `GAMMA` | float | LR decay factor for the MultiStepLR scheduler |
| `Iteration` | int | Total reconstruction iterations |
| `MAX_ITERATION` | int | L-BFGS inner iterations per step |
| `NUM_RESTARTS` | int | How many random restarts to try |
| `TV_WEIGHT` | float | Total variation regularization weight (0.0 = off) |
| `GRAD_LOSS` | str | `'cos'` (cosine similarity) or `'l2'` |
| `COMPUTE_JACOBIAN_RANK` | bool | Compute Jacobian rank before/after masking |
| `JACOBIAN_MAX_ENTRIES` | int | Cap on gradient entries used for Jacobian (reduces OOM) |
| `num_dummy` | int | Batch size to reconstruct (usually 1) |

## 5. Gradient Masking Modes

All masking is implemented in `functions/masking.py::build_gradient_mask()`. The function returns either `(keep_ids, None)` for tensor-wise masking or `(None, entry_masks)` for element-wise masking.

| Mode | What it does |
|------|-------------|
| `gradsize_topk` | Keep the K tensors (parameter tensors) with the largest gradient magnitude |
| `gradsize_topfrac` | Keep the top fraction of tensors by gradient magnitude |
| `gradsize_topk_entries` | Keep the K individual gradient scalar entries (across all tensors) with largest magnitude |
| `gradsize_topfrac_entries` | Keep the top fraction of scalar entries across all tensors |
| `gradsize_topk_entries_layer` | Per-tensor independent top-K entries — each parameter tensor keeps its own top K |
| `gradsize_topfrac_entries_layer` | Per-tensor independent top-fraction entries — each tensor keeps its own top fraction. For torchvision VGG models, all `classifier.*` layers are excluded from this global mode (they dominate parameter count and slow `create_graph=True`); label inference still works because it reads the original unmasked FC gradient. Use the `prefix_*_entries_layer` modes to include specific classifier sub-layers. |
| `prefix` | Keep only tensors whose parameter name starts with one of the given prefixes |
| `prefix_topk` | Within prefix-matched tensors, keep top-K by magnitude |
| `prefix_topfrac` | Within prefix-matched tensors, keep the top fraction by magnitude |
| `prefix_topk_entries` | Within prefix-matched tensors, keep top-K scalar entries globally |
| `prefix_topfrac_entries` | Within prefix-matched tensors, keep the top fraction of scalar entries globally |
| `prefix_topk_entries_layer` | Within prefix groups, keep top-K scalar entries per layer |
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
7. The **iDLG label recovery trick** recovers the true label from the gradient of the final fully-connected layer's weight: `label = argmin(sum(fc_weight_grad, dim=-1))`, computed once from the original *unmasked* gradient. This is exact for single-sample batches.
8. Best reconstruction snapshots are selected by gradient-matching loss.

**Loss functions:**
- `cos` — `1 - cosine_similarity(dummy_grad_flat, true_grad_flat)` — more robust, default
- `l2` — MSE between gradient vectors

---

## 7. Known Quirks and Gotchas

- **`Iteration` vs `num_iterations`:** Config uses uppercase `Iteration` (not `num_iterations`). This is inconsistent with Python convention and easy to mistype.
- **Normalized reconstruction space:** Dummy images are optimized in normalized
  pixel space and converted back to `[0, 1]` for metrics and saving. If you add
  a dataset with different normalization, wire its stats into
  `functions/consts.py` and the display logic.
- **Normalization constants:** `functions/consts.py` is the single source of truth for the main codebase. `invertinggradients/inversefed/consts.py` is a separate copy inside the submodule — do not modify it.
- **`invertinggradients/` is a submodule:** It has its own `.git`. Do not commit files inside it to the main repo. If you need to update it: `cd invertinggradients && git pull`.
- **Plotting convention:** Statistical plots use seaborn (`sns.lineplot`, `sns.barplot`) with matplotlib only for figure creation, labels, saving, and image rendering (`imshow`). Reconstruction panels and GIF frames remain matplotlib image displays because seaborn does not replace RGB image rendering.
- **Matplotlib backend:** `helper/visualization.py` line 2 forces `matplotlib.use("Agg")` for headless operation. Do not call `matplotlib.pyplot` before this runs in any file that uses visualization.
- **HPC C++ runtime path:** After activating the conda environment on DTU HPC, run `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"` before Python. Without this, compiled SciPy/Matplotlib extensions may load the old system `/lib64/libstdc++.so.6` and fail with `CXXABI_1.3.15 not found`. Add the export to job scripts before the Python command.
- **Optional plotting fallback:** `iDLG_mask.py` lazy-loads visualization helpers. If Matplotlib/Seaborn cannot import, experiments continue without PNG panels, GIFs, or restart plots; registry JSON and CSV outputs are still written so plots can be generated later.
- **Optional SciPy stats fallback:** `functions/io_utils.py` lazy-loads `scipy.stats` only when paired comparisons are needed. If SciPy cannot import, paired p-values and Shapiro-Wilk normality checks are skipped and the CI uses a normal-approximation critical value. Fix the HPC library path before producing final statistical results.
- **HPC scripts:** `scripts/` targets the DTU HPC cluster (LSF job scheduler). The `init.sh` sets up the conda environment from `environment.yml`.
- **`run_single_exp.py` is a worker module:** Run experiments through
  `iDLG_mask.py`; it builds the internal config dict and spawns workers.
- **FC is for label inference only, not forced into reconstruction:** `build_gradient_mask` does NOT preserve the last `nn.Linear` layer. The iDLG label trick is computed once in `run_single_exp.py` from the original *unmasked* final FC weight gradient (before masking), so label recovery always works. The FC/classifier gradient enters the reconstruction loss only if the chosen mask mode/prefixes select it. This makes FC/classifier ablations meaningful: excluding FC from the prefix list really excludes it from the objective. The Jacobian sweep can drop FC entirely with `--exclude_fc`; `rank_reconstruction_plot` has its own `--keep_fc` flag (select from non-FC entries only).
- **LFW normalization constants:** Computed manually in `archive/testing/compute_lfw_stats.py` and hardcoded in `functions/consts.py`. If you change the LFW preprocessing (resize, crop), recompute these.
- **Masking sweep workflow:** Run normal `iDLG_mask.py` commands for each `--gradsize_topfrac`; all runs with the same config (same network/dataset/optimizer/lr/etc.) append to the same CSV in `results/masking_sweeps/` regardless of `--run_id` or `--num_exp`. Then call `helper/plot_masking_sweep_csv.py <sweep_csv>`; its default reconstruction threshold is `--threshold_mse 0.01`. The plot script includes the matching iDLG baseline as the 0% masked point when `results/baselines/idlg_baselines_registry.json` contains the corresponding baseline key.
- **Jacobian dtype comparisons:** `functions/jacobian_rank_sweep.py --both_dtypes` executes the same sweep for `float32` and `float64`, writes one CSV with a `dtype` column, and saves one overlaid plot. If `--qr_pivot` is also enabled, the QR-pivot lines are dashed and use the same color as the corresponding dtype.
- **VGG fraction sweeps:** For `gradsize_topfrac_entries_layer` on VGG, all `classifier.*` layers (`classifier.0/3/6.*`) are excluded automatically, which substantially reduces VGG sweep runtime and memory use. Label inference is unaffected (it reads the original unmasked `classifier.6` gradient). To include `classifier.6` (or any specific classifier layer) in the reconstruction, switch to `prefix_topfrac_entries_layer` and list it in `--prefixes`.
- **Row normalisation in rank sweep:** Default is **no normalisation** — the relative threshold reflects actual gradient magnitudes. Pass `--normalise` to L2-normalise rows before `matrix_rank`. If you enable normalisation, verify rank is non-decreasing across row counts as a sanity check.
- **SVD diagnostics:** `--print_svd_info` now prints `sigma_max`, the computed `atol`, whether normalisation was applied, and the bottom-10 singular values. Use this to check whether borderline singular values have comfortable margin above the threshold.

---

## 8. Current Implementation Notes

- `iDLG_mask.py` is the high-level runner: CLI parsing, worker config construction,
  scheduling, output ordering, and summary printing.
- `run_single_exp.py` owns reconstruction behavior.
- `functions/experiment_results.py` owns aggregation, paired summaries, CSV row
  construction, and masking-sweep CSV writes.
- `helper/plot_masking_sweep_csv.py` plots registry-backed masking sweeps.
- `helper/plots.py` plots layer-ablation PSNR confidence intervals.
- Statistical charts use Seaborn. Matplotlib remains the backend for axes,
  saving, and image rendering.
- `iDLG_mask.py` can continue without plots when plotting libraries fail to
  import. Fix the HPC library path before producing final outputs.

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

Requires CUDA 12.8. On DTU HPC, load modules first (see `scripts/init.sh`) and
prepend `$CONDA_PREFIX/lib` to `LD_LIBRARY_PATH`. The active runner requires at
least one CUDA GPU.

---

## 11. Output Interpretation

**PSNR (Peak Signal-to-Noise Ratio):** Higher = better reconstruction. Typical values:
- > 30 dB: visually close to original
- 20–30 dB: recognizable but noisy
- < 20 dB: poor reconstruction / attack fails

**SSIM (Structural Similarity):** 0–1, higher = better. Correlates with perceptual quality better than PSNR.

**Jacobian rank:** The rank of `d(gradient_vector) / d(image_pixels)`. Full rank = gradient contains maximum information about the image. Rank drops as masking removes more gradient tensors. Used to quantify *why* masking degrades reconstruction.

**CSV output columns** (masked/both rows): `method`, `timestamp`, `job_id`, `dataset`, `network`, `restarts`, `lr`, `iteration`, `num_exp`, `tv_weight`, `optimizer`, `max_iter`, `history`, `mask_mode`, `prefixes`, `grad_param`, `med_best_loss`, `avg_best_loss`, `med_best_mse`, `avg_best_mse`, `avg_best_psnr`, `std_best_psnr`, `avg_best_ssim`, `std_best_ssim`, `mse_ci`, `mse_significant`, `psnr_ci`, `psnr_significant`, `psnr_normality`, `mse_normality`, `png_path`, `registry_key`

`psnr_normality` / `mse_normality` — Shapiro-Wilk result string on paired differences, e.g. `"normal (W=0.9821, p=0.3412)"` or `"NON-NORMAL (W=0.8123, p=0.0031)"`. Empty for iDLG-only rows.
