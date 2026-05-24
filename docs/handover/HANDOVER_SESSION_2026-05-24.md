# Session Handover — 2026-05-24

## Changes this session

### Normality results added to final summary print — `iDLG_mask.py`

The paired MSE and PSNR lines in the end-of-run summary now include the Shapiro-Wilk result:

```
Paired best PSNR diff (masked - iDLG): 2.341 dB | 95% CI: [...] | significant: True, masked | normality: normal (W=0.982, p=0.341)
Paired best MSE  diff (masked - iDLG): -0.0000123 | 95% CI: [...] | significant: True, masked | normality: NON-NORMAL (W=0.812, p=0.003)
```

---

### Code review fixes — `849651c`

A full codebase review was run. All findings resolved:

**Critical:**
- `functions/io_utils.py:84–88` — `paired_summary` early-return was missing `"normality_str"` key. When any experiment crashed (n=1), a `KeyError` aborted the entire results phase — no CSV was written. Fixed by adding `"normality_str": "n/a"` to the early-return dict.
- `functions/masking.py:346–350` — `prefix_topfrac` was routing through global (`get_keep_ids_by_gradsize`) instead of per-prefix ranking (`get_keep_ids_by_prefix_group`), silently inconsistent with `prefix_topk`. Fixed to match.

**Important:**
- `run_single_exp.py:168` — `final_weight_idx = len(original_dy_dx) - 2` was a fragile positional assumption that breaks if any network has `bias=False` on its last FC layer. Now derived properly via `_get_last_fc_param_indices(net)`.
- `iDLG_mask.py:772–775` — `png_path` column in CSV was hardcoded to the HPC prefix `/work3/s234843/bachelor/results/...` regardless of where the run happened. Now uses the actual `save_path` variable.
- Dead `USE_INVERSEFED_IDLG` config key removed from `iDLG_mask.py`.
- Stale help text fixed: `--iteration` no longer references non-existent early-stopping params; `--dataset` no longer references non-existent `--data_path` arg.

**Minor:**
- `helper/metrics.py:16` — PSNR denominator had spurious `+eps` introducing a small negative bias. Removed.
- `helper/visualization.py:41–47` — dead `set_xlabel` block (invisible after `ax.axis('off')`) removed.
- `run_single_exp.py:432–433` — dead `unsqueeze(0)` branch removed from SSIM computation (shape is always 4D).

---

### Deleted stale jacobian_parallel files — `b071e61`

Both `archive/jacobian_parallel.py` and `archive/jacobian_parallel_functions.py` deleted. Both were superseded by `functions/jacobian_rank_sweep.py` and had broken imports (`build_network`, `lfw_dataset` — both removed in earlier refactors).

---

### MSE threshold visualisation script — `helper/visualise_mse_threshold.py`

New standalone script for calibrating the "reconstructed" threshold used in the masking-sweep graph (% gradients masked vs. number of images reconstructed below threshold).

**What it does:** Runs baseline iDLG on `--num_exp` images and saves:

| Output | Purpose |
|--------|---------|
| `sorted_mse.png` | MSEs sorted ascending — natural break points visible as jumps |
| `mse_histogram.png` | Distribution of best-MSE values |
| `recon_grid.png` | GT + reconstruction sorted by MSE, labeled with MSE; green = below threshold, red = above |
| `mse_results.csv` | `rank`, `idx`, `best_mse`, `best_psnr` [, `reconstructed` if `--threshold` given] |

**Workflow:**
```bash
# Step 1 — run without threshold, inspect sorted_mse.png and recon_grid.png
python helper/visualise_mse_threshold.py --network resnet18 --dataset cifar100 --num_exp 30

# Step 2 — run with chosen threshold; CSV gains 'reconstructed' column
python helper/visualise_mse_threshold.py --network resnet18 --dataset cifar100 --num_exp 30 --threshold 0.03
```

**Key args:** `--network`, `--dataset`, `--pretrained`, `--num_exp`, `--iteration` (default 300), `--lr` (default 1.0), `--num_restarts` (default 1), `--tv_weight`, `--device`, `--run_id`, `--threshold`.

Uses cosine similarity loss and LBFGS, consistent with the main experiment pipeline. `_get_last_fc_param_indices` used for label inference (architecture-independent).

---

---

### MSE visualisation sweep + calibration — `helper/masking_sweep.py` + `iDLG_mask.py`

New feature for producing the "% gradients masked vs. images reconstructed" graph, with integrated threshold calibration so the full workflow runs from `iDLG_mask.py` without a separate script.

**`helper/masking_sweep.py`** — module with shared internals and two public functions:

| Function | Purpose |
|---|---|
| `_setup(args, ...)` | Shared: builds network, criterion, dm/ds/lb/ub |
| `_run_one(..., mask_mode=None, topfrac=None, return_images=False)` | Shared: one iDLG experiment; no masking when `mask_mode=None`; supports lbfgs/adam/adamw, cos/l2 loss, TV, restarts |
| `run_mse_calibration(...)` | Baseline iDLG on N images → sorted_mse.png, recon_grid.png, mse_results.csv |
| `run_mse_sweep(...)` | Masked sweep over `gradsize_topfrac_entries` and `gradsize_topfrac_entries_layer` → sweep_results.csv, sweep_plot.png |

No code is duplicated between calibration and sweep — both call `_setup` and `_run_one`.

**`iDLG_mask.py`** — two new args:

| Arg | Type | Purpose |
|---|---|---|
| `--mse_visualise` | flag | Switch into calibration or sweep mode (skips normal experiment) |
| `--threshold_mse` | float | MSE threshold; omit for calibration pass, supply for sweep pass |

Routing: `--mse_visualise` alone → `run_mse_calibration`; `--mse_visualise --threshold_mse X` → `run_mse_sweep`. `signed_adam/signed_adamw` rejected at the routing point. Runs immediately after `load_dataset()`, returns before multiprocessing — no CSV/panel output.

**Two-step workflow (entirely from `iDLG_mask.py`):**
```bash
# Step 1 — calibrate: inspect sorted_mse.png + recon_grid.png to pick a threshold
python iDLG_mask.py --network resnet18 --dataset cifar100 \
    --num_exp 30 --iteration 300 --mse_visualise

# Step 2 — sweep with chosen threshold
python iDLG_mask.py --network resnet18 --dataset cifar100 \
    --num_exp 20 --iteration 300 \
    --mse_visualise --threshold_mse 0.05
```

All other `iDLG_mask.py` args (`--lr`, `--optimizer`, `--grad_loss`, `--tv_weight`, `--num_restarts`, `--run_id`, `--pretrained`, `--gamma`) are passed through and respected by both modes.

Step 1 output saved to `results/threshold_<network>_<dataset>_<timestamp>/`.
Step 2 output saved to `results/sweep_<network>_<dataset>_<timestamp>/`.

---

### Paired-list filtering bug fix — `iDLG_mask.py`

**Root cause:** MSE and PSNR paired lists were built by two independent `if` blocks. An experiment where PSNR = +inf (MSE < 1e-12, near-perfect reconstruction) was dropped from the PSNR list but kept in the MSE list, giving Shapiro-Wilk different sample sizes for the two metrics — producing a spuriously high PSNR normality p-value (0.8 instead of 0.012 in a confirmed case with 28 vs 30 points).

**Fix:** both `METHODS == "both"` and `METHODS == "masked"` paths now use a single joint validity check across all four values (mse_idlg, mse_masked, psnr_idlg, psnr_masked). An experiment is either included in both lists or excluded from both, with a warning printed. The fix is in `iDLG_mask.py` lines ~596–612 and ~657–672.

---

### Shapiro-Wilk normality check — confirmed correct (after fix)

The Shapiro-Wilk test in `functions/io_utils.py:62` is applied to `d = x_masked - x_idlg` (the paired differences) — correct for a paired t-test.

**High PSNR p-values / low MSE p-values are a genuine statistical property, not a bug.** PSNR = −10·log₁₀(MSE) is a log transform — a standard variance-stabilising technique that makes positively-skewed MSE differences more normal. MSE differences are bounded at 0 and right-skewed; PSNR differences are approximately normal. Both tests now always run on the same n experiments.

**Note on low power:** with n < 30, Shapiro-Wilk has very low power. "normal (p=0.8)" means the test cannot detect non-normality at this sample size — not that the data is definitively normal.

---

## Decisions carried forward

- The LBFGS closure is called twice per iteration intentionally (matches the original iDLG paper).
- `prefix_topfrac` now correctly does per-prefix ranking — any prior results using this mode should be treated as potentially inconsistent with `prefix_topk` and re-run.
