# Session Handover — 2026-05-28

---

## Changes (Claude session)

### `layer_spread`: group by architecture stage, expand VGG-style Sequentials — `8ed283d`

**Files:** `helper/metrics.py`

Added `_get_layer_groups(net)` helper that determines group labels for `layer_spread` by inspecting the network's module hierarchy rather than splitting parameter name strings.

**Behaviour by architecture:**

- **ResNet-18/50/101**: top-level children are used directly as groups — `conv1`, `bn1`, `layer1`, `layer2`, `layer3`, `layer4`, `fc` (7 groups). `layer1`–`layer4` contain `BasicBlock` / `Bottleneck` modules which have their own sub-modules, so they are not expanded.

- **VGG-11/13/16**: `features` is a `nn.Sequential` whose direct children are all leaf modules (`Conv2d`, `ReLU`, `MaxPool2d`). It is expanded one level: each conv layer becomes its own group — `features.0`, `features.2`, … `features.28` for VGG-16 (13 conv groups). `classifier` is similarly expanded to `classifier.0`, `classifier.3`, `classifier.6` (3 linear groups). Total: 16 groups for VGG-16.

**Rule:** a `nn.Sequential` is expanded if and only if all of its direct children are leaf modules (no sub-modules). Otherwise the Sequential itself is one group.

The `layer_spread` branch now calls `_get_layer_groups` instead of `name.split('.')[0]`. A fallback to `name.split('.')[0]` is retained for any parameter not found in the mapping.

---

### `layer_spread` bug fix: redistribute surplus budget from capped groups — `ef077c2`

**Files:** `helper/metrics.py`

**Bug:** After dividing the k-entry budget evenly across groups and capping each group at its actual size, the surplus from small groups was silently discarded. For ResNet-18 with `--max_row_count 10000`:
- `bn1` group has 128 entries (weight + bias, both 1-D) but received an allocation of 1428
- After `min(1428, 128)`, the surplus of 1300 entries was dropped
- J_max had shape `(8700, 3072)` instead of `(10000, 3072)`
- All row count sweep values ≥ 8700 returned the same 8700-row Jacobian

**Fix:** After capping, a redistribution loop passes the leftover budget to groups with remaining capacity (largest groups first):

```python
leftover = k - sum(per_group.values())
if leftover > 0:
    for grp in order:  # largest-first
        if per_group[grp] < group_totals[grp]:
            give = min(leftover, group_totals[grp] - per_group[grp])
            per_group[grp] += give
            leftover -= give
            if leftover == 0:
                break
```

For ResNet-18 at k=10000, the 1300 surplus from `bn1` is now absorbed by `layer4` (8.4 M entries). J_max is now exactly `(10000, 3072)` as requested. Verified for k ∈ {500, 1000, 5000, 10000}.

**Understanding the `qr_pivot` rank curve:** with `layer_spread + qr_pivot`, the results show `rank = k` for all k ≤ 3072, then plateau at 3072. This is correct and expected — QR pivot with column pivoting on J_max^T selects rows in order of maximum linear independence, guaranteeing `rank = min(k, rank(J_max))`. For a well-initialised ResNet-18 on a 32×32 image the Jacobian pool has full column rank (3072), so QR pivot always finds k independent rows up to the 3072 ceiling.

---

## Changes (Mathias, same day)

### Interactive job queue — `ff4c644`, `f5c52d3`

**Files:** `scripts/interactive_jobscript.sh`

New job script for submitting interactive experiments to the HPC queue. Cleaned up comments in a follow-up commit.

---

### Hotfix: crash on file write — `f893ce7`

**Files:** `functions/io_utils.py`, `helper/masking_sweep.py`, `helper/visualization.py`, `iDLG_mask.py`, `tests/test_safe_io.py`

Wrapped all file-write operations in safe I/O helpers to prevent partial writes / crashes mid-run from corrupting result files. Added `tests/test_safe_io.py` with 20 tests covering the safe-write helpers. Total test count is now 45 (was 25).

---

### SSIM added to CSV and registry — `0ed01e9`

**Files:** `functions/io_utils.py`, `iDLG_mask.py`

SSIM is now tracked alongside PSNR and MSE throughout the run pipeline:

- `update_masked_registry` and `update_idlg_baseline` now accept and store `best_ssim_list`
- `write_baseline_summary_csv` now emits `avg_best_ssim`, `std_best_ssim`, `best_ssim_list` columns
- `paired_summary` handles `metric="ssim"` (higher-is-better direction)

---

### Abort run on single worker failure — `170ef1f`

**Files:** `iDLG_mask.py`

Previously, a failed worker (GPU crash, OOM, etc.) printed an error and the run continued with missing results. Now a `_abort_run` closure terminates all active processes and calls `sys.exit(1)` on the first failure, ensuring no partial result files are written.

---

## Test status

45 tests pass (`tests/test_masking.py` 25 + `tests/test_safe_io.py` 20).

---

## Changes (Claude session, same day)

### `jacobian_rank_sweep.py`: always include `unknowns` in row counts — `d031240`

**Files:** `functions/jacobian_rank_sweep.py`

After `row_counts` is finalised (from either `--row_counts` or `--stepsize`), `unknowns` (= C×H×W) is inserted if not already present and the list is re-sorted. Guarantees a data point at exactly the theoretical minimum row count regardless of step alignment.

---

### `--min_row_count`: configurable sweep start, defaults to `unknowns` floored to nearest thousand — `01d54ab`

**Files:** `functions/jacobian_rank_sweep.py`

New argument `--min_row_count` sets where the stepsize range begins. Default when omitted: `(unknowns // 1000) * 1000` — e.g. 3000 for CIFAR (unknowns=3072). Combined with the `unknowns` insertion above, the default sweep for CIFAR with `--stepsize 100` starts `[3000, 3072, 3100, 3200, ...]` instead of `[100, 200, ...]`.

Validation: `--min_row_count` must be positive and must not exceed `--max_row_count`.

---

### Jacobian build and SVD on GPU — `b4c217e`

**Files:** `functions/jacobian_rank_sweep.py`, `helper/metrics.py`

`device_for_J` in `_worker_core` changed from the hardcoded `"cpu"` to `device` (the actual compute device). J is now built and stored on GPU, giving significant speedup for the fwAD passes.

`_qr_rank` (and the QR pivot path in `compute_jacobian_rank_sweep`) calls `.cpu()` before `.numpy()` so scipy can consume the array regardless of where J lives.

---

### `matrix_rank` runs on CPU (LAPACK) to avoid cuSOLVER warnings — `2cf5dc7`

**Files:** `helper/metrics.py`

`_rank_of_J` now calls `J.cpu()` before row-normalisation and `matrix_rank`. This eliminates `UserWarning: torch.linalg.svd: During SVD computation with the selected cusolver driver, batches 0 failed to converge` which appeared when J was on GPU and cuSOLVER hit a convergence issue. The result was always correct (PyTorch auto-falls back), but the warning flooded logs and the fallback path is slower. CPU LAPACK is reliable for float64 SVD.

J still lives on GPU during fwAD construction; only the SVD step moves to CPU.

---

### Progress bar tracks J build passes and rank SVDs separately — `9957323`

**Files:** `functions/jacobian_rank_sweep.py`, `helper/metrics.py`

**Before:** the tqdm bar was frozen during J construction (the slow part) and only updated once per row count after all SVDs were done.

**After:**
- `_build_jacobian` accepts `progress_fn=None` and calls it after every fwAD / backward pass, giving a smoothly moving bar during J construction.
- `compute_jacobian_rank_sweep` accepts `j_progress_fn` (forwarded to `_build_jacobian`) and `rank_progress_fn` (called after each SVD).
- `total_steps` = `n_samples × (j_passes + rank_passes)` where `j_passes = unknowns` (fwAD) or `sum(min(k, unknowns))` (independent mode) and `rank_passes = len(row_counts) × (2 if qr_pivot else 1)`.
- Per-sample header and summary lines written via `tqdm.write(file=sys.stderr)` so they don't corrupt the bar.

---

### `--independent`: theoretically correct per-k Jacobian rank — `dbd3fbd`

**Files:** `helper/metrics.py`, `functions/jacobian_rank_sweep.py`

**Problem:** the default build-once approach builds J at `max(row_counts)` and slices `J_max[:k]` for each smaller k. For `layer_spread`, the pool composition changes with the total budget — so rank at k=4000 from a 10000-row pool differs from rank at k=4000 from a 5000-row pool. Results are not comparable across runs with different `--max_row_count`.

**Fix:** `--independent` builds J separately for each k using `max_entries=k`. Rank at k=4000 always reflects exactly what an attacker gets when receiving 4000 gradient entries selected by `--jacobian_select_mode`. Results are deterministic and independent of `--max_row_count`.

**Cost:** `len(row_counts)` × more forward passes. With 20 row counts, use `--stepsize 500` or `--stepsize 1000` to keep runtime manageable.

**`qr_pivot` in independent mode:** QR-pivoting J_k (which already has exactly k rows) and taking all k rows gives the same rank as J_k — so `--qr_pivot` is redundant with `--independent`. It remains supported for interface consistency but adds no information.

**CSV:** `independent` column added so results files are self-documenting.

**Recommended command for thesis results:**
```bash
python functions/jacobian_rank_sweep.py \
    --network resnet18 --dataset cifar100 \
    --stepsize 500 --max_row_count 10000 \
    --num_samples 8 --num_workers 4 \
    --jacobian_select_mode layer_spread \
    --independent
```

---

### `--sample_indices`: target specific dataset indices — `0dfc155`

**Files:** `functions/jacobian_rank_sweep.py`

New argument `--sample_indices` accepts a comma-separated list of dataset indices and runs the sweep on exactly those images, bypassing the random permutation. Overrides `--num_samples` and `--run_id`.

```bash
python functions/jacobian_rank_sweep.py \
    --network resnet18 --dataset cifar100 \
    --stepsize 500 --max_row_count 10000 \
    --jacobian_select_mode layer_spread \
    --independent \
    --sample_indices 23784
```

Use case: a run produced an anomalous rank (e.g. rank=2975 instead of 3072) for a specific image. `--sample_indices` lets you reproduce and investigate that single image without rerunning the full batch.

---

## Changes (second Claude session, same day)

### Fix: MSE sweep (`--mse_visualise --threshold_mse`) now matches main experiment behaviour

**Files:** `helper/masking_sweep.py`

**Problem:** `_run_one` in the sweep captured MSE only at the **final iteration** of each restart, then took the minimum across restarts. The main experiments (`run_single_exp.py`) capture MSE at the **best-gradient-loss iteration** within each restart. Because a run can peak early and then drift, the two approaches give different MSE values for the same experiment — so the calibration threshold and sweep counts were not comparable to main experiment results.

Additionally, the lbfgs path in `_run_one` always clamped `dummy`, whereas the main code skips clamping for LeNet/LeNet_bigger (to avoid corrupting the quasi-Newton approximation).

**Fix:** Within each restart's iteration loop, `restart_best_loss` and `restart_best_x` now track the snapshot at the lowest gradient loss, identical to the main code. MSE is then computed from that snapshot, not from the final-iteration state. The LeNet clamping exception was also added to the lbfgs path.

**Scope of match:** After this fix the sweep matches the main experiments exactly for ResNet/CIFAR. The only remaining minor difference is that the sweep computes `autograd.grad` over all network parameters then masks post-hoc, while the main code restricts autograd to the selected parameter subset — numerically equivalent but slightly less efficient for heavily masked runs.

**What this means for results:** Calibration thresholds and sweep reconstruction counts are now directly comparable to main experiment PSNR/MSE values. Re-run calibration if you intend to compare sweep output against previously collected main experiment data.
