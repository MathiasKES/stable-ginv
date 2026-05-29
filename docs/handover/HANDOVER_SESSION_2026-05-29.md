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

**Files:** `iDLG_mask.py`, `scripts/plot_masking_sweep_csv.py`, `helper/masking_sweep.py`

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

New script:

```bash
python scripts/plot_masking_sweep_csv.py \
  results/masking_sweeps/<sweep_csv>.csv \
  --threshold_mse 0.03
```

The plotting script loads `results/baselines/masked_registry.json` by default, resolves each row's `masked_key`, reads `best_mse_list`, and computes threshold counts from the registry. It also loads `results/baselines/idlg_baselines_registry.json` by default, derives the matching baseline key from the masked registry args, and includes the baseline as the 0% masked point when present. Outputs:
- `masking_sweep_summary.csv`
- `sweep_plot.png`
- `sweep_bar.png`

Pass `--registry_path <path>` or `--baseline_registry_path <path>` only if the registries are not in the default `results/baselines/` location relative to the sweep CSV. Pass `--no_baseline` to omit the baseline point.

### Main results CSV now links to registries

`exp_results_<network>.csv` now includes `registry_key` as the final column:
- `iDLG` rows store the baseline registry key.
- `iDLG_masked` rows store the masked registry key.
- `--methods both` writes both rows with their respective keys.
