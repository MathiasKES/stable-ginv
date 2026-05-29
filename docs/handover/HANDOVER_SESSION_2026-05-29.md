# Session Handover — 2026-05-29

---

## Changes (Claude session)

### Fix: MSE sweep now matches main experiment MSE criterion — `7c2cee3`

**Files:** `helper/masking_sweep.py`

The sweep (`--mse_visualise --threshold_mse`) previously captured MSE only at the **final iteration** of each restart, then took the minimum across restarts. The main experiments (`run_single_exp.py`) capture MSE at the **best-gradient-loss iteration** within each restart.

Fix: within each restart's iteration loop, `restart_best_loss` and `restart_best_x` now track the snapshot at the lowest gradient loss (identical to the main code). MSE is computed from that snapshot, not from the final-iteration state. The LeNet/LeNet_bigger clamping exception was also added to the lbfgs path (main code skips clamping for LeNet to avoid corrupting the quasi-Newton approximation).

**Practical implication:** Calibration thresholds and sweep reconstruction counts are now directly comparable to main experiment MSE/PSNR values. Re-run calibration if comparing sweep output against previously collected main experiment data.

**Remaining minor difference:** The sweep computes `autograd.grad` over all network parameters then masks post-hoc, while the main code restricts autograd to the selected parameter subset. Numerically equivalent, slightly less efficient for heavily masked runs.

---

### Add k=3 to restart gain CI in restart curve CSV — `b775abd`

**Files:** `iDLG_mask.py`

The restart curve CSV previously only computed paired-difference confidence intervals (vs k=1) for k=5 and k=10. Added k=3 to the list, so the k=3 row now also gets `gain_psnr_*`, `ci_low_psnr_*`, `ci_high_psnr_*`, `normality_psnr_*` (and MSE equivalents) appended.

Change: `gain_ks = [k for k in [5, 10] if k <= NUM_RESTARTS]` → `[3, 5, 10]`.

---

## Notes

- The non-monotonic PSNR across restarts in the restart figure is **expected behaviour**, not a bug. The best restart is selected by gradient matching loss, not pixel MSE. A restart with lower gradient loss can produce a worse reconstruction — this means gradient loss is an imperfect proxy for pixel quality, which is itself an interesting finding. No code change was made.
