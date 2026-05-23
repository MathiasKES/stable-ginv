# Handover: Developer / Researcher

**Audience:** A developer or researcher who is continuing active work on this codebase — adding new masking modes, running new experiments, debugging, or extending the pipeline.

**Last updated:** 2026-05-23

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
│   ├── io_utils.py          Baseline registry, paired stats, CSV helpers
│   ├── Dataset.py           Dataset loaders: MNIST, CIFAR-10/100, LFW
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
Iterates over a grid of masking parameters and logs rank results.

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
| `GRADSIZE_TOPFRAC` | float | 0.0–1.0, fraction to keep (for `gradsize_topfrac`) |
| `GRADSIZE_THRESHOLD` | float or None | Magnitude threshold (for `gradsize_threshold`) |
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
| `gradsize_threshold` | Keep tensors whose gradient magnitude exceeds a threshold |
| `gradsize_topk_entries` | Keep the K individual gradient scalar entries (across all tensors) with largest magnitude |
| `gradsize_topfrac_entries` | Keep the top fraction of scalar entries across all tensors |
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

---

## 8. Recent Changes (as of 2026-05-23)

- `functions/masking.py` — last FC layer now always preserved in `build_gradient_mask` (tensor-wise: force-unioned into keep_ids; entry-wise: all-True mask override); `_grad_magnitude` helper extracted to remove duplicated metric block; `get_keep_ids` prefix branch now delegates to `get_prefix_keep_ids`
- `helper/Network.py` — `get_model` now handles all architectures including `LeNet`, `LeNet_bigger`, `MediumCNN`, `BiggerCNN`; no need to call `build_network` separately
- `helper/training_utils.py` — `build_network` deleted; only `make_scheduler` remains
- `run_single_exp.py`, `functions/jacobian_rank_sweep.py` — updated to import `get_model` from `helper.Network` directly
- `tests/test_masking.py` — 21 tests (up from 19); 2 new tests verify last-FC invariant
- `run_single_exp.py` — 85-line inline masking dispatch removed; now calls `build_gradient_mask()` from `functions/masking.py`
- `functions/Dataset.py` — `Dataset_from_Image` renamed to `_Dataset_from_Image` (private, only used inside `lfw_dataset()`)
- `functions/Misc_functions.py` — deleted (was already a dead re-export shim)

From git log:
- `45ee353` — Added SSIM metric; added per-image SSIM and PSNR logging
- `8707bbd` — L-BFGS now forces original iDLG hyperparameters (lr=1, iter=300, max_iter=20)
- `1bfa981` — Removed gradient normalization when using L-BFGS optimizer
- `ea7bd4e` — Fixed Jacobian rank script for new prefix masking setup; default restarts = 1
- `50059fe` — Computed and hardcoded LFW normalization constants
- `61de43c` — Implemented LFW dataset support
- `c26a71c` — Gamma parameter now included in output field names
- `123b1fa` — Made LR scheduler gamma an adjustable parameter

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

**CSV output columns:** `network`, `dataset`, `mask_mode`, `prefixes`, `PSNR`, `SSIM`, `loss`, `MSE`, `CI_lower`, `CI_upper`, `Significance`, `Optimizer`, `LR`, `Iterations`
