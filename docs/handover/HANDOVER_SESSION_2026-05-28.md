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
