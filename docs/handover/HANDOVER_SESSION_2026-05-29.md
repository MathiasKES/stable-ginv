# Session Handover — 2026-05-29

---

## Changes (Claude session)

### Fix: MSE sweep now matches main experiment MSE criterion — `7c2cee3`

**Files:** `helper/masking_sweep.py`

The sweep (`--mse_visualise --threshold_mse`) previously captured MSE only at the **final iteration** of each restart, then took the minimum across restarts. The main experiments (`run_single_exp.py`) capture MSE at the **best-gradient-loss iteration** within each restart.

Fix: within each restart's iteration loop, `restart_best_loss` and `restart_best_x` now track the snapshot at the lowest gradient loss (identical to the main code). MSE is computed from that snapshot, not from the final-iteration state. The LeNet/LeNet_bigger clamping exception was also added to the lbfgs path (main code skips clamping for LeNet to avoid corrupting the quasi-Newton approximation).

**Practical implication:** Calibration thresholds and sweep reconstruction counts are now directly comparable to main experiment MSE/PSNR values. Re-run calibration if comparing sweep output against previously collected main experiment data.

**Remaining minor difference:** The sweep computes `autograd.grad` over all network parameters then masks post-hoc, while the main code restricts autograd to the selected parameter subset. Numerically equivalent, slightly less efficient for heavily masked runs.

**Superseded later the same day:** The separate `helper/masking_sweep.py` experiment path was subsequently removed. Use the registry-backed normal-run workflow described in the Codex section below.

---

### Add k=3 to restart gain CI in restart curve CSV — `b775abd`

**Files:** `iDLG_mask.py`

The restart curve CSV previously only computed paired-difference confidence intervals (vs k=1) for k=5 and k=10. Added k=3 to the list, so the k=3 row now also gets `gain_psnr_*`, `ci_low_psnr_*`, `ci_high_psnr_*`, `normality_psnr_*` (and MSE equivalents) appended.

Change: `gain_ks = [k for k in [5, 10] if k <= NUM_RESTARTS]` → `[3, 5, 10]`.

---

## Notes

- The non-monotonic PSNR across restarts in the restart figure is **expected behaviour**, not a bug. The best restart is selected by gradient matching loss, not pixel MSE. A restart with lower gradient loss can produce a worse reconstruction — this means gradient loss is an imperfect proxy for pixel quality, which is itself an interesting finding. No code change was made.

---

## Changes (Codex session)

### Simplify masking sweep results flow — `62988ad`

**Files:** `iDLG_mask.py`, `helper/plot_masking_sweep_csv.py`, `helper/masking_sweep.py`

The old experiment-side sweep mode was removed:
- `--mse_visualise` removed from `iDLG_mask.py`
- `--threshold_mse` removed from `iDLG_mask.py`
- `--sweep_step` removed from `iDLG_mask.py`
- `helper/masking_sweep.py` deleted

The new workflow uses normal experiment commands. For example:

```bash
python iDLG_mask.py \
  --network vgg13 \
  --mask_mode gradsize_topfrac_entries_layer \
  --gradsize_topfrac 0.10 \
  --num_exp 30 --iteration 5000 \
  --optimizer signed_adamw --lr 0.1 \
  --grad_loss cos --num_restarts 1
```

If `--mask_mode gradsize_topfrac_entries_layer` is used without explicitly passing `--methods`, the runner switches from the default `idlg` to `masked`, so the masked run and sweep row are produced.

Each normal masked run still writes its full result to `results/baselines/masked_registry.json`. For `gradsize_topfrac_entries_layer`, the runner also appends a compact row to a config-specific CSV in `results/masking_sweeps/` with only:
- `command`
- `topfrac`
- `masked_key`

The sweep CSV filename hashes the masked-registry comparable args with `gradsize_topfrac` removed, so all fractions for the same setup append to the same file.

### Plotting now reads the masked registry

Plotting utility:

```bash
python helper/plot_masking_sweep_csv.py \
  results/masking_sweeps/<sweep_csv>.csv
```

The plotting script loads `results/baselines/masked_registry.json` by default, resolves each row's `masked_key`, reads `best_mse_list`, and computes threshold counts from the registry. The default threshold is `--threshold_mse 0.01`; pass a different value only for a deliberate sensitivity check. It also loads `results/baselines/idlg_baselines_registry.json` by default, derives the matching baseline key from the masked registry args, and includes the baseline as the 0% masked point when present. Outputs:
- `masking_sweep_summary.csv`
- `sweep_plot.png`
- `sweep_bar.png`

Current naming now appends network, dataset, and threshold to those files, e.g. `sweep_plot_vgg13_cifar100_threshold_0p01.png`.

Pass `--registry_path <path>` or `--baseline_registry_path <path>` only if the registries are not in the default `results/baselines/` location relative to the sweep CSV. Pass `--no_baseline` to omit the baseline point.

### Main results CSV now links to registries

`exp_results_<network>.csv` now includes `registry_key` as the final column:
- `iDLG` rows store the baseline registry key.
- `iDLG_masked` rows store the masked registry key.
- `--methods both` writes both rows with their respective keys.

### Add dtype comparison to Jacobian rank sweep — `cfed65e`

**Files:** `functions/jacobian_rank_sweep.py`

The Jacobian rank sweep now supports dtype control:

```bash
python functions/jacobian_rank_sweep.py --dtype float32
python functions/jacobian_rank_sweep.py --dtype float64
python functions/jacobian_rank_sweep.py --both_dtypes
```

Default remains `float64`, preserving the previous behaviour. `--both_dtypes` runs the same sweep twice, once with `float32` and once with `float64`, reseeding before each run so the same randomly initialized model/sample choices are used for comparison. The output CSV includes a `dtype` column, and the plot overlays both dtype curves in one graph with separate colors. If `--qr_pivot` is enabled, QR-pivot curves are drawn dashed in the same color as their dtype's select-mode curve.

### Exclude VGG classifier head from per-layer frac masking

**Files:** `functions/masking.py`, `tests/test_masking.py`

For torchvision VGG models using `--mask_mode gradsize_topfrac_entries_layer`, the per-layer entry mask now excludes the intermediate classifier layers by default:

- `classifier.0.*` is removed.
- `classifier.3.*` is removed.
- `classifier.6.weight` and `classifier.6.bias` are still fully kept by the existing last-FC label-inference rule.
- `features.*` still gets the requested per-tensor top fraction.

Example for `vgg13 --gradsize_topfrac 0.5`: each `features.*` parameter keeps its top 50% entries, `classifier.0` and `classifier.3` are not included in gradient matching, and `classifier.6` is fully visible. This keeps iDLG label recovery intact while avoiding the huge VGG classifier tensors that dominate runtime and memory in fraction sweeps.

---

## Changes (Claude session, second pass — 2026-05-29)

### Jacobian rank sweep refactor — not yet committed at session start

**Files:** `functions/jacobian_rank_sweep.py`, `helper/metrics.py`

These changes were present locally but not yet pushed to `origin/main`. No behavioural changes; pure cleanup and two targeted fixes.

**`functions/jacobian_rank_sweep.py` — helper extraction:**
Inline blocks repeated across `main()` and `_mp_worker` were extracted into named helpers:

| Helper | Replaces |
|---|---|
| `_storage_root()` | Duplicated HPC path check |
| `_save_dir()` | Inline results path logic |
| `_seed_all(seed)` | 3-line torch/numpy/cuda seed block (was copy-pasted 3×) |
| `_resolve_device(device, worker_rank)` | Inline device string logic in two places |
| `_new_rank_results(row_counts, qr_pivot)` | Dict initialisation for rank accumulators |
| `_summarize_rank_results(...)` | Mean/std computation loop |
| `_print_rank_summary(...)` | Per-sample rank print loop |
| `_run_serial(...)` | Serial execution path |
| `_run_parallel(...)` | Parallel execution path |
| `_run_dtype(...)` | Full per-dtype body (~90 lines in `main`) |

Also: `_print_mask_debug()` had two unused arguments (`prefixes`, `prefix_layer_fracs`) that were removed.

**`helper/metrics.py` — two fixes:**
- `selected_idx.to(x_norm.device)` moved outside the fwAD column loop (was called once per column, i.e. 3072× per sample for CIFAR). Small but real speedup.
- `_qr_rank()` renamed to `_qr_pivot_rows()` (returns reordered J only; rank computation separated). In independent mode, `--qr_pivot` now skips the redundant QR + SVD since reordering an already independently-built J_k cannot change its rank.

---

### Print rank for all row counts, not just max — unpushed

**Files:** `functions/jacobian_rank_sweep.py`

`_worker_core` previously only printed the rank line when processing the maximum row count. The `if rows == max_rows:` guard was removed so a rank line is emitted for every row count per sample:

```
[1/30] rows=4000: rank=2891  shape=(4000, 3072)
[1/30] rows=5000: rank=2961  shape=(5000, 3072)
...
[1/30] rows=10000: rank=3072  shape=(10000, 3072)
```

---

### `--no_normalisation`: optional row normalisation toggle — unpushed

**Files:** `functions/jacobian_rank_sweep.py`, `helper/metrics.py`

By default, `_rank_of_J` divides each row of J by its L2 norm before calling `matrix_rank`. This was added previously to prevent a monotonicity bug (rank decreasing with more rows) caused by large-magnitude rows inflating `sigma_max` and thus the adaptive threshold.

New flag `--no_normalisation` skips this step. Without normalisation, `matrix_rank` uses its default relative threshold (`max(M,N) * eps * sigma_max`) directly on the raw J. This means the rank reflects gradient *magnitude-weighted* independence rather than pure *directional* independence — arguably more relevant to whether the attacker can actually reconstruct an image, since very small gradient entries carry negligible signal.

**Risk:** the monotonicity bug (rank non-increasing with k) can return without normalisation if rows span many orders of magnitude. Verify rank is non-decreasing across your `--row_counts` when using this flag.

Implementation: `normalize_rows=True` parameter added to `_rank_of_J` and `compute_jacobian_rank_sweep`; `not args.no_normalisation` is passed through from `_worker_core`.

---

### Enhanced `--print_svd_info` output — unpushed

**Files:** `helper/metrics.py`

The SVD info line previously only printed the bottom-10 singular values. It now also prints `sigma_max`, the computed `atol`, and whether normalisation was applied:

```
[SVD] M=6000 N=3072  normalised=True  sigma_max=4.21e+01  atol=1.38e-12  bottom-10: [...]
```

This makes it easy to see whether the bottom singular values have comfortable margin above the threshold or are borderline.

---

## Changes (Claude session — 2026-05-30)

### Fix masking sweep CSV grouping — `a7cef13`

**Files:** `iDLG_mask.py`

Previously, the sweep CSV filename hash included `run_id` and `num_exp` (via `masked_comparable_args`). Since these differ between job submissions, each run produced a unique hash → one CSV per run, rather than one CSV per config. Runs with the same setup but different topfrac values were not being grouped together.

Fix: `run_id`, `num_exp`, and `gradsize_topk` are now excluded from the hash before it is computed. The CSV filename now depends only on the structural config (network, dataset, optimizer, lr, gamma, grad_loss, iteration, tv_weight, num_restarts, mask_mode, gradsize_metric, prefixes). All topfrac values for the same config now append to the same file regardless of batch size or random seed.

**Compatibility:** Existing CSVs on the HPC were named with the old hash and will not match new runs. The underlying data is preserved in `masked_registry.json` and can be re-plotted with `helper/plot_masking_sweep_csv.py`.

---

### Add `sample_indices` column to Jacobian rank sweep CSV — `a7cef13`

**Files:** `functions/jacobian_rank_sweep.py`

The CSV output now includes a `sample_indices` column (semicolon-separated dataset indices) in the same order as `per_sample_ranks`. This makes it possible to identify which dataset image produced an anomalous rank (e.g. rank=2904 vs 3072) without needing to reproduce the random seed manually.

---

## Changes (Codex session — 2026-05-31)

### DTU HPC compiled-library resilience — `645519b`, `ad80623`

DTU HPC Python processes were loading the old system `/lib64/libstdc++.so.6` instead of the conda environment library. Compiled SciPy and Matplotlib extensions then failed with:

```text
version `CXXABI_1.3.15' not found
```

Permanent jobscript fix after conda activation:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

Code resilience changes:
- `functions/io_utils.py` lazy-loads `scipy.stats`. If unavailable, experiment startup continues; paired p-values and Shapiro-Wilk normality checks are skipped, and CI calculation falls back to a normal-approximation critical value.
- `iDLG_mask.py` lazy-loads visualization helpers. If Matplotlib/Seaborn cannot import, experiments continue without PNG panels, GIFs, or restart plots while registry JSON and CSV outputs are still written.
- `safe_savefig()` logs plot-save errors instead of aborting the run.
- The initial no-plot fallback buffer bug was fixed in `ad80623`.
