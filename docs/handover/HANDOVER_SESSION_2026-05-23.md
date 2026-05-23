# Session Handover — 2026-05-23

## Phase 1 status (as of this session)

Most of Phase 1 was completed in earlier sessions. Current state:

| Item | Status |
|------|--------|
| P1.1 Split `Misc_functions.py` | ✅ Done — split into `functions/masking.py`, `functions/io_utils.py`, `helper/metrics.py`, `helper/training_utils.py`, `helper/visualization.py`; `functions/Misc_functions.py` deleted |
| P1.2 Deduplicate normalization constants | ✅ Done — all stats centralised in `functions/consts.py` |
| P1.3 Remove commented-out code | ✅ Done — 6 debug lines deleted from `iDLG_mask.py` |
| P1.4 Docstrings on new modules | ✅ Done |
| P1.5 Delete `original/` | ✅ Done — moved to `archive/` |
| P1.6 Move `weights_init` | ✅ Done — canonical home is `helper/Network.py` |

## Phase 1 complete

All Phase 1 items are done. **P2.4 unit tests also completed this session (see below).**

Recommended remaining Phase 2 order: P2.1 config dataclasses, P2.2 shared experiment core, P2.3 type hints.

---

## Phase 2 progress (this session)

### P2.4 Unit tests — ✅ DONE

Created `tests/test_masking.py` with 19 pytest tests (all pass in ~1 second on CPU):

| Group | Tests | What is covered |
|-------|-------|-----------------|
| `flatten_observed_gradients` | 5 | all grads, keep_ids filter, entry_masks filter, None skip, raises on empty keep_ids |
| `get_keep_ids_by_gradsize` | 5 | topk, topfrac, threshold, threshold fallback, None gradients ranked last |
| `get_entry_masks_by_gradsize` | 4 | topk entries, topfrac, shape preservation, candidate_ids restriction |
| `build_gradient_mask` routing | 5 | idlg keeps all, gradsize_topk, gradsize_topfrac, topfrac_entries, prefix mode |

Supporting files added: `conftest.py` (repo root, empty — ensures pytest path detection), `tests/__init__.py` (empty).

Run with: `python -m pytest tests/ -v`

---

### Code cleanup: dispatch consolidation in `run_single_exp.py`

`run_single_exp.py` contained ~85 lines of inline `if/elif` masking dispatch that duplicated `build_gradient_mask()` in `functions/masking.py`. This was the only entry point not using the central dispatcher — meaning any new masking mode added to `build_gradient_mask` would silently have no effect when using `run_single_exp.py`.

**What changed:**
- Import changed from 6 individual masking functions to `from functions.masking import build_gradient_mask`
- Entire inline dispatch block (lines ~155–239) replaced with:
  ```python
  keep_ids, entry_masks = build_gradient_mask(
      method="idlg" if method == "iDLG" else "masked",
      mask_mode=MASK_MODE,
      net=net,
      original_dy_dx=original_dy_dx,
      prefixes=PREFIXES,
      prefix_layer_fracs=PREFIX_LAYER_FRACS,
      gradsize_topk=GRADSIZE_TOPK,
      gradsize_topfrac=GRADSIZE_TOPFRAC,
      gradsize_threshold=GRADSIZE_THRESHOLD,
      gradsize_metric=GRADSIZE_METRIC,
  )
  if entry_masks is not None:
      observed_entries = sum(int(m.sum().item()) for m in entry_masks if m is not None)
  else:
      observed_entries = sum(
          g.numel() for i, g in enumerate(original_dy_dx)
          if g is not None and i in keep_ids
      )
  ```

`jacobian_rank_sweep.py` already called `build_gradient_mask` correctly — no changes needed there.

---

### `Dataset_from_Image` made private

Renamed `Dataset_from_Image` → `_Dataset_from_Image` in `functions/Dataset.py`. The class is only used internally by `lfw_dataset()` in the same file — it was never part of the public API.

---

### Key behavioral note: `gradsize_topfrac` vs `gradsize_topfrac_entries`

These two modes are easy to confuse:

- **`gradsize_topfrac=0.5`** keeps the top 50% of **parameter tensors** by L2 norm. On VGG13, the FC layers (`classifier.0.weight`: ~102M params) dominate parameter count and always have high L2 norms. Keeping 50% of tensors by norm typically preserves ~99.96% of all scalar entries — the masking effect on total gradient size is minimal.
- **`gradsize_topfrac_entries=0.5`** (i.e. `--mask_mode gradsize_topfrac_entries --gradsize_topfrac 0.5`) keeps the top 50% of **individual scalar gradient entries** globally. This gives `kept_fraction ≈ 0.50` as expected.

If the goal is to control the fraction of gradient *information* shared, use `gradsize_topfrac_entries`.

---

---

## Additional cleanup (same session, later)

### Last FC layer always preserved

`build_gradient_mask` now enforces that the last `nn.Linear` layer's parameters are never masked, regardless of mode:
- Tensor-wise (`keep_ids`): last FC indices are unioned into `keep_ids` after the mode's selection runs
- Entry-wise (`entry_masks`): last FC entries are overridden to an all-True mask after the mode's selection runs

Helper `_get_last_fc_param_indices(net)` walks `net.named_modules()` to find the last `nn.Linear`, then maps its weight/bias to their parameter indices.

Two new tests added (`test_last_fc_always_preserved_in_topk`, `test_last_fc_always_preserved_in_entry_masks`).

### `functions/masking.py` internal deduplication

- `_grad_magnitude(g, metric)` extracted — removes the identical 6-line metric block that existed in both `get_keep_ids_by_gradsize` and `get_keep_ids_by_prefix_group`
- `get_keep_ids` prefix branch now calls `get_prefix_keep_ids` instead of re-implementing the same loop

### Model factory consolidated into `get_model`

`get_model` in `helper/Network.py` now handles all architectures: `LeNet`, `LeNet_bigger`, `MediumCNN`, `BiggerCNN` (custom CNNs) and all torchvision backbones. `build_network` in `helper/training_utils.py` was deleted; `training_utils.py` now contains only `make_scheduler`. Callers (`run_single_exp.py`, `functions/jacobian_rank_sweep.py`) updated to import `get_model` from `helper.Network` directly.

---

---

## jacobian_rank_sweep.py sync + gradsize_threshold removal

### `jacobian_rank_sweep.py` synced with main scripts

`jacobian_rank_sweep.py` lagged behind `iDLG_mask.py` and `run_single_exp.py` in several ways. All gaps closed:

- **Prefix parsing** — replaced a 20-line manual loop with `PREFIXES, PREFIX_LAYER_FRACS = parse_prefixes_with_fracs(args.prefixes)` (shared utility in `functions/io_utils.py`)
- **Dataset loading** — now uses `load_dataset(args.dataset, data_path)` (shared utility in `functions/Dataset.py`)
- **Normalization** — was always using dataset-specific stats even for pretrained networks. Fixed: when `--pretrained` and `channel == 3`, uses `consts.imagenet_mean / imagenet_std`; otherwise uses `{dataset}_mean / {dataset}_std`
- **weights_init** — was applied to all non-resnet networks. Fixed: only applied to custom CNNs (`LeNet`, `LeNet_bigger`, `MediumCNN`, `BiggerCNN`) and only when not pretrained

### `gradsize_threshold` mode removed everywhere

The `gradsize_threshold` mask mode was never used in practice and was removed:
- Removed `--gradsize_threshold` argparse argument from `iDLG_mask.py` and `jacobian_rank_sweep.py`
- Removed `gradsize_threshold=None` parameter from `build_gradient_mask()`
- Removed `elif mask_mode == "gradsize_threshold"` branch from `build_gradient_mask()`
- Removed `GRADSIZE_THRESHOLD` from config dict and `run_single_exp.py`

### New per-layer entry modes

`gradsize_topfrac_entries_layer` and `gradsize_topk_entries_layer` added to `build_gradient_mask`. These apply the fraction/topk independently to each parameter tensor, so the overall gradient distribution is preserved across layers. Implemented as a sub-case of `get_entry_masks_by_prefix_group` with every param name as its own prefix — no new function needed.

### Shared utility functions extracted

| Function | Location | Purpose |
|----------|----------|---------|
| `load_dataset(dataset, data_path)` | `functions/Dataset.py` | Returns `(dst, channel, num_classes, shape_img)` |
| `parse_prefixes_with_fracs(prefixes_str)` | `functions/io_utils.py` | Parses `"conv1:0.5,layer1:1.0,fc"` → `(tuple, dict)` |

Both `iDLG_mask.py` and `jacobian_rank_sweep.py` now use these instead of inline logic.

### Test count

24 tests total (up from the 21 after last-FC work, 5 new per-layer mode tests added).

---

## Decisions carried forward

No new decisions this session — see `HANDOVER_SESSION_2026-05-22.md` for all prior context.

---

---

## Second session (2026-05-23, continued)

### TV normalization fix — matches Geiping et al.

`run_single_exp.py` was computing TV on `x_raw = (dummy_data * ds + dm).clamp(0,1)` — de-normalized [0,1] space. The paper computes TV on `x_trial` (the normalized image, same space the network sees — equivalent to `dummy_data` after sigmoid). Fix: remove the `x_raw` line from both the L-BFGS closure and the Adam path; pass `dummy_data` directly to `total_variation()`.

The `current_x = (dummy_data.detach() * ds + dm).clamp(0.0, 1.0)` line at the bottom of the loop was **not touched** — it is only used for MSE computation and GIF frame capture, not for the loss.

**TV weight note:** The paper's DEFAULT_CONFIG lists `0.1`, but actual single-image experiments with trained networks use `0.01`. Use `--tv_weight 0.01` for fair comparison with paper results.

---

### `functions/jacobian_rank_sweep.py` — serial + parallel merged

`archive/jacobian_parallel.py` (parallel-only) and `functions/jacobian_rank_sweep.py` (serial-only) were merged into one file. New structure:

- `_worker_core(args, sample_indices, device, row_counts, prefixes, prefix_layer_fracs, progress_fn=None, debug=False)` — shared computation, returns `{rows: [rank_list]}`
- `_mp_worker(...)` — `mp.spawn` target that wraps `_worker_core` with a progress counter
- `main()`: `--num_workers 1` (default) calls `_worker_core` directly with tqdm; `--num_workers 2–4` uses `mp.spawn`

Per-sample diagnostics added inside `_worker_core`:
```python
print(f"  sample idx={idx}: grad_norm={grad_norm:.4f}  nan={has_nan}  inf={has_inf}")
# ... at max row count:
print(f"    max_rows={rows}: rank={jac_rank}  shape={jac_shape}")
```

---

### Jacobian rank monotonicity fix — `helper/metrics.py`

**Problem:** rank declined as row count increased. Root cause: `torch.linalg.matrix_rank` default `rtol` threshold = `max(M,N) · ε · σ_max`. As rows grow, σ_max grows (O(M)), so the threshold grows as O(M²·ε), eventually swamping genuine singular values.

**Fix:**
1. Row-normalize J before SVD: `J_norm = J / row_norms.clamp_min(1e-30)`. After normalization σ_max ≤ √M.
2. Use a fixed absolute threshold: `atol = max(rank_tol, max(M,N) · ε · M^0.5)` with `rtol=0.0`.
3. For float64 with M ≤ 50000, the precision floor ≈ 2.5e-9 — well below `rank_tol=1e-6`, so `rank_tol` dominates in practice.

The `normalize_rows` parameter (was always False, never used) was removed from the signature and all call sites.

---

### Shapiro-Wilk normality test — `functions/io_utils.py`

`paired_t_ci` now returns `"shapiro_stat"` and `"shapiro_p"` (NaN when n < 3).

`paired_summary` now returns `"normality_str"`:
- `"n/a"` when n < 3
- `"normal (W=..., p=...)"` when Shapiro-Wilk p > 0.05
- `"NON-NORMAL (W=..., p=...)"` when p ≤ 0.05

---

### Masked registry — `functions/io_utils.py` + `iDLG_mask.py`

New functions in `io_utils.py` (mirrors the iDLG baseline registry pattern):

```python
masked_key_from_args(args)          # MD5 hash of all hyperparams (reconstruction + masking)
load_masked_registry(path)          # returns {} if file not found
save_masked_registry(path, registry)
update_masked_registry(registry, key, comparable_args, best_psnr_list, best_mse_list)
```

`iDLG_mask.py` saves a masked registry entry at `results/baselines/masked_registry.json` after each masked run. The key covers all reconstruction and masking hyperparameters; overwriting an existing key triggers a warning.

CSV now includes a `psnr_normality` column (Shapiro-Wilk result string for the PSNR differences).

---

### Example run command

```bash
python iDLG_mask.py \
  --network resnet18 --dataset cifar100 \
  --methods both \
  --mask_mode gradsize_topfrac_entries \
  --gradsize_topfrac 0.5 \
  --num_exp 30 --iteration 5000 \
  --optimizer signed_adamw --lr 0.1 \
  --grad_loss cos --tv_weight 0.01 \
  --num_restarts 1 --run_id 0
```

This will:
- Run 30 paired experiments (iDLG baseline + masked with entry-wise top-50%)
- Save iDLG baseline to `results/baselines/idlg_baselines_registry.json`
- Save masked results to `results/baselines/masked_registry.json`
- Print Shapiro-Wilk normality test result to stdout
- Write `psnr_normality` to the CSV output
