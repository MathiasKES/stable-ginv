# Session Handover — 2026-05-25

## Changes this session

### Merge conflict resolved — `e48f3ff`

Remote (`da4a387`) had introduced `exp_idx` print info in the old sequential dispatch body. Our HEAD had already refactored that body into the `_handle_result` closure. Resolved by keeping HEAD (closure already incorporated `exp_idx`) and discarding the duplicate old accumulation block from the remote side.

---

### Parallel restart dispatch — `4426165`

When `num_exp < num_gpus` AND `NUM_RESTARTS > 1`, restarts run in parallel across GPUs instead of sequentially within each worker.

**Mechanism:**
- Builds task list `[(exp_i, r_i) for exp_i in range(num_exp) for r_i in range(NUM_RESTARTS)]`
- Each task dispatched with `config['SINGLE_RESTART_IDX'] = r_i`; `run_single_exp` then runs only that one restart
- Results buffered in `restart_buf[exp_i][r_i]`
- After all tasks complete: `_merge_restart_results(restart_buf[exp_i])` called per experiment

**`_merge_restart_results`** (defined in `iDLG_mask.py`):
- Sorts restarts by index, picks the best iDLG result (min `best_mse_iDLG`) and best masked result independently
- Builds running-best PSNR, image, MSE, and SSIM lists via `_running_best_*` helpers
- Returns a merged dict with the same shape as a normal single-experiment result

**`run_single_exp.py`:**
- `SINGLE_RESTART_IDX = config.get('SINGLE_RESTART_IDX')` — when set, restart loop runs only that index via `_restart_range = [SINGLE_RESTART_IDX] if SINGLE_RESTART_IDX is not None else range(NUM_RESTARTS)`
- Result dict includes `restart_idx` field

**Seed formula:** `restart_seed = seed * 1000 + restart_idx` (unchanged from sequential mode — seeds are deterministic and consistent).

---

### Per-restart tracking — `run_single_exp.py`

Four new per-restart lists tracked and returned in result dict:

| Key | Content |
|---|---|
| `psnr_per_restart_idlg/masked` | Running-best PSNR after each restart (list length `NUM_RESTARTS`) |
| `img_per_restart_idlg/masked` | Running-best reconstruction image (numpy `(1,C,H,W)`) after each restart |
| `mse_per_restart_idlg/masked` | Running-best MSE after each restart |
| `ssim_per_restart_idlg/masked` | SSIM of running-best reconstruction after each restart |

All four are "running best" — `list[k]` is the best value seen across restarts 0..k, not just restart k.

---

### Restart curve: CSV + plot — `decec5e` / `4426165`

Saved when `NUM_RESTARTS > 1`. Files include timestamp to avoid overwriting:

- `restart_curve_<timestamp>.csv` — columns: `num_restarts`, `mean_psnr_idlg`, `std_psnr_idlg`, `mean_psnr_masked`, `std_psnr_masked`; rows k=5 and k=10 additionally include gain+CI columns (see below)
- `restart_curve_<timestamp>.png` — mean best PSNR ± std vs. number of restarts used

---

### Restart image figure — `decec5e` → `bcee6f4` → `61af180`

Saved when `NUM_RESTARTS > 1`. File: `restart_images_<timestamp>.png`.

**Columns:** k=1, k=3, k=5, k=10 (any column where `k > NUM_RESTARTS` is silently skipped).

**Rows:** GT / iDLG / masked (depending on `METHODS`).

**Representative image when `num_exp > 1`:** picks the experiment whose PSNR at `max(display_ks)` is closest to the median across all experiments. Falls back to index 0.

**Annotations under each reconstruction cell:** `MSE: X.XXXXX`, `PSNR: XX.XX dB`, `SSIM: X.XXX`.

---

### PSNR + MSE gain table with 95% paired CI — `964410f`

Printed to console after the restart curve is saved. Also written as extra columns in `restart_curve_<timestamp>.csv` on rows k=5 and k=10.

Reuses `paired_t_ci` from `functions/io_utils.py:69` — same function used for the overall masked-vs-iDLG paired summary.

**Pairing:** for each image `i`, gain at k = `metric[i][k-1] - metric[i][0]`. The CI is over the distribution of per-image gains.

**Console format:**
```
Restart gain summary vs k=1 (95% paired CI):
           PSNR gain k=5              MSE gain k=5    ...
  iDLG     +X.XXX [L, H] p=0.XXX    -X.XXXXX [L, H] p=0.XXX ...
           normality: normal (W=0.9821, p=0.341)      normality: NON-NORMAL (W=0.812, p=0.003) ...
  masked   ...
```

Normality uses the same Shapiro-Wilk format as the existing paired masked-vs-iDLG summary (`normal` if p > 0.05, `NON-NORMAL` otherwise). Applied to the vector of per-image gain differences.

**CSV columns added at rows k=5 and k=10:** `gain_psnr_idlg`, `ci_low_psnr_idlg`, `ci_high_psnr_idlg`, `normality_psnr_idlg`, `gain_mse_idlg`, `ci_low_mse_idlg`, `ci_high_mse_idlg`, `normality_mse_idlg` (and masked variants).

---

## Recommended workflow for thesis results

```bash
# Statistically sound restart analysis (20 images × 10 restarts):
python iDLG_mask.py --network resnet18 --dataset cifar100 \
    --num_exp 20 --iteration 300 --num_restarts 10
```

This produces:
1. `restart_curve_<ts>.png/.csv` — mean ± std curve + gain table with 95% CIs
2. `restart_images_<ts>.png` — 4-column figure (k=1,3,5,10) of the median-PSNR representative image with MSE/PSNR/SSIM annotations

**Note on statistical interpretation:**
- PSNR gains tend to be approximately normal (log transform stabilises variance)
- MSE gains are often non-normal (right-skewed, bounded at 0)
- With n < 30, Shapiro-Wilk has low power — "normal (p=0.8)" means the test cannot detect non-normality, not that the data is definitively normal

---

## Decisions carried forward

- `parallel_restarts` is automatic: triggered when `num_exp < num_gpus AND NUM_RESTARTS > 1`. No flag needed.
- The restart image figure always uses fixed columns `[1, 3, 5, 10]` — not configurable at runtime, change `display_ks` in `iDLG_mask.py` if needed.
- Gain table always computes at k=5 and k=10 vs k=1. Skipped gracefully if `NUM_RESTARTS < 5` or `< 10`.
