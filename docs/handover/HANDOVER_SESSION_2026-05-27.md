# Session Handover — 2026-05-27

---

## Changes (fourth batch, same day)

### `layer_spread` rewritten: spread evenly across layer groups, not individual tensors

**Files:** `helper/metrics.py`

The previous `layer_spread` implementation spread the budget across individual parameter tensors (~62 for ResNet-18). With a budget of 20000 this gave only ~490 entries per tensor. For deep conv layers with millions of parameters, 490 entries is too sparse to contribute meaningful rank — `layer_spread` was actually worse than `topk_abs` (rank ceiling of ~2631 vs full rank 3072).

**New behaviour:** budget is spread evenly across **layer groups** identified by the first component of the parameter name — e.g., `conv1`, `bn1`, `layer1`, `layer2`, `layer3`, `layer4`, `fc` for ResNet-18. Within each group, the top-magnitude entries are selected globally across all tensors in that group.

For ResNet-18 with `--max_row_count 20000`, each of the 7 groups gets ~2857 entries. `conv1` and `bn1` are smaller than 2857 so they are capped and the surplus goes to the largest groups.

This allows the most informative entries from each architectural stage to be selected, rather than wasting the budget on tiny BN tensors.

---

### `layer_spread` fix: 1D tensors included in full, budget on weight tensors — `dfd12c9`

**Files:** `helper/metrics.py`

Before the layer-group rewrite, an intermediate fix was applied: 1D parameter tensors (BN scale/shift, biases) were included in their entirety without consuming the spread budget. The budget was then spread evenly across weight tensors only (ndim ≥ 2). This prevented BN clamping from reducing the actual selected count below the requested `max_entries`. The layer-group rewrite supersedes this fix.

---

## Changes (third batch, same day)

### `--stepsize` and `--max_row_count` for automatic row count generation — `a6d88c8`

**Files:** `functions/jacobian_rank_sweep.py`

Instead of manually writing out every value in `--row_counts`, specify a step size and upper bound. Row counts are generated as `[stepsize, 2×stepsize, ..., max_row_count]`. If `max_row_count` is not exactly divisible by `stepsize`, it is appended as the final value.

```bash
# replaces: --row_counts 1000,2000,3000,...,10000
python functions/jacobian_rank_sweep.py --stepsize 1000 --max_row_count 10000
```

`--row_counts` still works unchanged when `--stepsize` is omitted. `--max_row_count` is required when `--stepsize` is used.

---

### `layer_spread` and `qr_pivot` Jacobian select modes — `69a9a21`

**Files:** `helper/metrics.py`, `functions/jacobian_rank_sweep.py`

Two new values for `--jacobian_select_mode`:

**`layer_spread`** — distributes the row budget evenly across parameter tensors (layers), then takes the top-magnitude entries within each layer's quota. This prevents `topk_abs` from concentrating all selected rows in a few deep layers and ensures early-layer (high-resolution, localised) Jacobian rows are always represented. Within-layer ordering is still by magnitude.

**`qr_pivot`** was originally added as a select mode but was then refactored into a standalone flag (see below). Do not use `qr_pivot` as a value for `--jacobian_select_mode`.

---

### `--qr_pivot` as a standalone flag — `da46fc7`

**Files:** `helper/metrics.py`, `functions/jacobian_rank_sweep.py`

`--qr_pivot` is now a boolean flag independent of `--jacobian_select_mode`. When set, after `J_max` is built using the chosen select mode, its rows are reordered via QR decomposition with column pivoting on `J_max^T`. `pivots[i]` is the index of the i-th most linearly independent row — so `J_max[pivots[:k]]` contains the best possible k rows from the pool.

This can be combined with any select mode:

```bash
--jacobian_select_mode topk_abs --qr_pivot      # magnitude pool, optimal ordering
--jacobian_select_mode layer_spread --qr_pivot  # diverse pool, optimal ordering (recommended)
--jacobian_select_mode layer_spread             # diverse pool, layer-then-magnitude ordering
```

Requires `scipy` (`pip install scipy`). The QR is applied to the already-built J (no extra forward/backward passes).

---

### Dual rank output when `--qr_pivot` — `feb5626`

**Files:** `helper/metrics.py`, `functions/jacobian_rank_sweep.py`

When `--qr_pivot` is used the sweep now computes and saves ranks for **both** orderings in a single run:

- `mean_rank` / `std_rank` / `per_sample_ranks` — select mode ordering (unchanged)
- `mean_rank_qr` / `std_rank_qr` / `per_sample_ranks_qr` — QR pivot ordering (new columns appended to CSV)

Stdout prints two separate per-sample rank tables labelled by mode. The plot shows both as separate curves on the same figure (solid = select mode, dashed = qr_pivot).

The QR pivot curve is always ≥ the base curve at every row count and both plateau at `unknowns` (3072 for CIFAR). The gap between them shows how much rank is "wasted" by ordering rows by magnitude rather than orthogonality.

---

### Row counts start at `stepsize`, not `unknowns` — `740209a`

**Files:** `functions/jacobian_rank_sweep.py`

When `--stepsize` is used, the generated range now starts at `stepsize` (not `unknowns`). This allows sweeping from low row counts regardless of network/dataset.

```bash
--stepsize 1000 --max_row_count 10000
# → [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
```

---

## Changes (second batch, same day)

### Per-sample ranks in CSV and stdout — `86a6522`

**Files:** `functions/jacobian_rank_sweep.py`

The sweep now records and prints the rank for every individual sample, not just the mean and std.

**CSV:** a `per_sample_ranks` column is appended to each row with a semicolon-separated list of integer ranks, one per sample:

```
rows_used,mean_rank,std_rank,unknowns,num_samples,jacobian_select_mode,per_sample_ranks
3072,3072.0,0.0,3072,3,topk_abs,3072;3072;3072
5000,4891.3,42.1,3072,3,topk_abs,4850;4920;4904
```

**Stdout:** printed after aggregation for every row count.

---

### Remove `--atol` and `--gradsize_metric` arguments — `eaa7ffb` / `86a6522`

**Files:** `functions/jacobian_rank_sweep.py`, `helper/metrics.py`

- `--atol` removed. `torch.linalg.matrix_rank` now uses its own default `rtol` (based on machine epsilon, matrix shape, and largest singular value). Applying a custom `atol` to the row-normalised Jacobian was misleading because the normalised matrix is dimensionless.
- `--gradsize_metric` removed. Gradient magnitude for tensor selection is always L2 norm (Frobenius norm). Exposing this as a flag added complexity without practical benefit.

Migration: remove `--atol` and `--gradsize_metric` from any existing job scripts.

---

### Jacobian rank sweep restructure: build J once — `eaa7ffb`

**Files:** `helper/metrics.py`, `functions/jacobian_rank_sweep.py`

Previously, `compute_jacobian_rank` was called separately for each row count in `row_counts`, so a sweep over 15 row counts meant 15 full forward+backward passes per sample. Now:

1. `_build_jacobian(net, x_norm, y, criterion, keep_ids, entry_masks, max_entries, select_mode, device_for_J)` — private helper that builds J at the requested `max_entries` size. Returns `(J, total_entries, unknowns)`.
2. `_rank_of_J(J, print_svd_info=False)` — row-normalises J and calls `matrix_rank`.
3. `compute_jacobian_rank_sweep(row_counts=...)` — calls `_build_jacobian` once at `max(row_counts)`, then slices `J_max[:k]` for each k and calls `_rank_of_J`.

Result: one forward+backward pass per sample regardless of how many row counts are swept.

`compute_jacobian_rank` (used by `run_single_exp.py`) is unchanged in interface — it now delegates to the same `_build_jacobian` + `_rank_of_J` helpers.

---

### Forward-mode AD for Jacobian columns — `57745ba`

**Files:** `helper/metrics.py`

`_build_jacobian` now automatically chooses the cheaper AD direction:

| Condition | Method | Passes |
|-----------|--------|--------|
| `unknowns < used_entries` | Forward-mode (column-wise) | `unknowns` passes |
| `unknowns >= used_entries` | Backward-mode (row-wise) | `used_entries` passes |

For a CIFAR image (3072 unknowns) with 30 000 gradient entries selected, forward-mode reduces the pass count from 30 000 to 3 072 — roughly **10× fewer passes**.

Implementation uses `torch.autograd.forward_ad` (`fwAD.dual_level`, `fwAD.make_dual`, `fwAD.unpack_dual`). Requires PyTorch ≥ 2.0 for `autograd.grad` to propagate dual tangents through the graph. The environment has PyTorch 2.8.0.

If `fwAD.unpack_dual(g_col).tangent` returns `None`, the code raises a `RuntimeError` with a clear message rather than silently producing zeros.

---

### Fixed mask debug display — `57745ba`

**Files:** `functions/jacobian_rank_sweep.py`

The old `_print_mask_debug` tried to match prefix names against parameter names and printed `kept 0/0` for every group when using a non-ResNet network (VGG, LeNet, etc.) because the prefix list defaulted to ResNet names.

The rewritten function:

- Iterates `net.named_parameters()` and partitions by whether the parameter index is in `keep_ids`.
- Prints the exact kept/skipped tensor names.
- Computes the kept entry fraction from `original_dy_dx` directly.
- Works correctly for any architecture.

Example output:
```
=== MASK DEBUG ===
mask_mode: gradsize_topfrac
entry_masks is None: True
num keep_ids: 14
observed_entries=14563840, total_entries=28327938, kept_fraction=0.514037
kept tensors (14): ['features.0.weight', 'features.0.bias', ...]
skipped tensors (14): ['features.2.weight', ...]
=== END MASK DEBUG ===
```

---

## Changes (first batch, 2026-05-27)

### Configurable MSE sweep step — `71f6cd7`

**Files:** `iDLG_mask.py`, `helper/masking_sweep.py`

Added `--sweep_step` for the MSE visualisation sweep.

Default remains `0.05`, preserving previous behavior:

```bash
python iDLG_mask.py --mse_visualise --threshold_mse 0.05
# topfrac: 0.05, 0.10, 0.15, ..., 1.0
```

Custom step example:

```bash
python iDLG_mask.py --mse_visualise --threshold_mse 0.05 --sweep_step 0.1
# topfrac: 0.1, 0.2, ..., 1.0
```

Validation:

```bash
python -m py_compile iDLG_mask.py helper/masking_sweep.py
python -m pytest tests/ -v
```

Result: 24 tests passed.

---

## Important usage note

`--sweep_step` only affects the masking-sweep graph when both of these are present:

```bash
--mse_visualise --threshold_mse <value>
```

If a command only includes `--sweep_step` without `--mse_visualise --threshold_mse`, it runs the normal experiment path and will not create `sweep_plot.png` or `sweep_results.csv`.

Normal experiment example, no sweep plot:

```bash
python iDLG_mask.py --methods both --mask_mode gradsize_topfrac_entries_layer \
    --gradsize_topfrac 0.9 --sweep_step 0.1
```

Sweep plot example:

```bash
python iDLG_mask.py --mask_mode gradsize_topfrac_entries_layer \
    --mse_visualise --threshold_mse 0.05 --sweep_step 0.1
```

`--methods` and `--gradsize_topfrac` are not used by the MSE sweep. The sweep sets `topfrac` internally from `--sweep_step` through `1.0`.

---

## Parallelism note

The MSE sweep still runs in parallel across GPUs. Parallelism is per sweep point:

1. Generate topfrac values from `--sweep_step` to `1.0`.
2. For each topfrac value, call `_run_parallel(...)`.
3. `_run_parallel(...)` distributes `--num_exp` experiments across all visible GPUs via one process per GPU slot.

The sweep points themselves are processed sequentially. For example, with 4 GPUs:

```bash
--num_exp 20 --sweep_step 0.1
```

runs 20 experiments across 4 GPUs at `topfrac=0.1`, then 20 across 4 GPUs at `topfrac=0.2`, and so on through `1.0`.
