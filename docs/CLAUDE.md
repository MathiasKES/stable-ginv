# stable-ginv — Codebase Map

## Claude instructions

- **Never commit automatically.** Only commit when the user explicitly asks.
- **Never add co-authors to commit messages.**

---

## Related repositories

`/home/mathias/GitHub/bachelor_thesis/` — Overleaf Git-integrated thesis. **Never auto-commit to this repo.** Changes must be committed manually by the user so Overleaf syncs correctly.

---

Research project: gradient inversion attacks on federated learning.
Compares baseline iDLG against masked variants (selective gradient disclosure) across multiple masking strategies.

---

## Quick orientation

**Entry points (project root)**

| File | Role |
|---|---|
| `iDLG_mask.py` | CLI entry point; argument parsing, dataset loading, multiprocess dispatch, CSV+PNG+GIF output |
| `run_single_exp.py` | Worker: one experiment on one GPU; gradient computation, optimization loop, metrics |

**`functions/` — core domain logic**

| File | Role |
|---|---|
| `functions/masking.py` | All gradient masking: `build_gradient_mask`, `get_keep_ids*`, `get_entry_masks*`, `flatten_observed_gradients` |
| `functions/io_utils.py` | Baseline registry, paired stats, CSV helpers, `parse_prefixes_with_fracs` |
| `functions/Dataset.py` | `load_dataset()` (shared loader), `lfw_dataset()`, `_Dataset_from_Image` (private) |
| `functions/consts.py` | Normalization constants: `{dataset}_mean`, `{dataset}_std` for cifar10, cifar100, mnist, imagenet |
| `functions/jacobian_rank_sweep.py` | Serial Jacobian rank sweep |

**`helper/` — shared utilities**

| File | Role |
|---|---|
| `helper/Network.py` | Model definitions + factory (`get_model`, `LeNet*`, `MediumCNN`, `BiggerCNN`) |
| `helper/metrics.py` | `compute_psnr_from_mse`, `compute_ssim_batch`, `total_variation`, `compute_jacobian_rank`, `compute_grad_match_loss` |
| `helper/training_utils.py` | `make_scheduler` only (`build_network` moved into `get_model`) |
| `helper/visualization.py` | `save_recon_panel`, `save_recon_gif` |

**`archive/`** — retired scripts (not imported anywhere): `iDLG_original.py`, `jacobian_parallel.py`, `run_single_exp_batch.py`, old visualize/testing scripts.

Paths: data → `./data` or `/work3/s234843/bachelor/datasets`; results → `./hpc/results` or `/work3/s234843/bachelor/results`. Detection is automatic via `os.access()`.

---

## Core algorithm

```
1. Sample GT image x from dataset
2. Forward/backward on frozen net → g_obs = ∇_θ CrossEntropy(net(x), label)
3. Apply mask → g_public  (subset of g_obs, same mask applied every step)
4. Init dummy x_d ~ N(0,1)
5. Optimize:  min_{x_d}  ||mask(∇_θ CE(net(sigmoid(x_d)), label_pred)) - g_public||² + λ·TV(x_d)
6. label_pred inferred from final FC layer gradient (iDLG trick); last FC is never masked (enforced by build_gradient_mask)
```

Restarts: repeat step 4–5 `NUM_RESTARTS` times, keep best by MSE.

---

## Masking modes (--mask_mode)

Two granularities: **tensor-wise** (keep whole parameter tensors) and **entry-wise** (keep individual scalar entries).

| Mode | Granularity | Selection |
|---|---|---|
| `gradsize_topk` | tensor | global top-K by metric |
| `gradsize_topfrac` | tensor | global top-fraction by metric |
| `gradsize_topk_entries` | entry | global top-K entries |
| `gradsize_topfrac_entries` | entry | global top-fraction entries |
| `gradsize_topk_entries_layer` | entry | per-tensor independent top-K entries |
| `gradsize_topfrac_entries_layer` | entry | per-tensor independent top-fraction entries |
| `prefix` | tensor | keep all tensors whose name starts with any prefix |
| `prefix_topk` | tensor | within prefix groups, top-K per group |
| `prefix_topfrac` | tensor | within prefix groups, top-fraction per group |
| `prefix_topk_entries` | entry | within prefix groups, top-K entries globally |
| `prefix_topfrac_entries` | entry | within prefix groups, top-fraction entries globally |
| `prefix_topk_entries_layer` | entry | within prefix groups, top-K entries **per layer** |
| `prefix_topfrac_entries_layer` | entry | within prefix groups, top-fraction entries **per layer** |

`--gradsize_metric`: `l2` (Frobenius norm), `mean_abs`, `sum_abs`.

`--prefixes`: comma-separated, optional per-layer fraction/k suffix: `conv1:1.0,layer1:0.5,fc:1.0`.

---

## Key functions — functions/masking.py

```python
# ── Masking entry points ──────────────────────────────────────────────────────
build_gradient_mask(method, mask_mode, net, original_dy_dx,
                    prefixes=(), prefix_layer_fracs=None,
                    gradsize_topk=20, gradsize_topfrac=0.5,
                    gradsize_metric='l2')
  -> (keep_ids: set|None, entry_masks: list[bool_tensor]|None)
# Central dispatcher. method='idlg' → keep_ids=all, no entry_masks.
# Invariant: last FC layer is ALWAYS included regardless of mask_mode —
# tensor-wise via keep_ids union, entry-wise via all-True mask override.

get_keep_ids_by_gradsize(original_dy_dx, mode, topk, top_frac,
                         metric, candidate_ids=None)
  -> (sorted keep_ids: list, sizes_sorted: list[(idx, score)])

get_keep_ids_by_prefix_group(net, original_dy_dx, prefixes,
                              mode, topk, top_frac,
                              prefix_top_fracs=None, prefix_top_ks=None,
                              metric='l2')
  -> (keep_ids: list, ranked_by_prefix: dict)

get_entry_masks_by_gradsize(original_dy_dx, mode, topk, top_frac,
                             candidate_ids=None)
  -> (entry_masks: list[bool_tensor|None], kept: int, total: int)

get_entry_masks_by_prefix_group(net, original_dy_dx, prefixes,
                                 mode, topk, top_frac,
                                 prefix_top_fracs=None)
  -> (entry_masks: list[bool_tensor|None], kept: int, total: int)

get_keep_ids(mask_mode, net=None, prefixes=None)  -> set
get_prefix_keep_ids(net, prefixes)                -> set

# ── Gradient utilities ────────────────────────────────────────────────────────
flatten_observed_gradients(grad_list, keep_ids=None, entry_masks=None) -> Tensor
# keep_ids and entry_masks are mutually exclusive; both None → flatten all.

compute_jacobian_rank(net, x_norm, y, criterion,
                      keep_ids=None, entry_masks=None,
                      max_entries=None, select_mode='topk_abs',
                      device_for_J='cpu', rank_tol=1e-6)
  -> (rank: int, shape: tuple, used_entries: int, unknowns: int)
# Row-by-row Jacobian build to avoid OOM.

# ── Metrics ───────────────────────────────────────────────────────────────────
compute_psnr_from_mse(mse, max_val=1.0) -> float   # 10·log₁₀(1/mse)
total_variation(x)                       -> scalar  # L1 TV regulariser

# ── Network factory ───────────────────────────────────────────────────────────
get_model(network, channel, num_classes, input_size, pretrained=False) -> nn.Module
# Handles ALL architectures: 'LeNet','LeNet_bigger','MediumCNN','BiggerCNN'
# and any torchvision backbone (resnet*, vgg*, wide_resnet*, densenet*).

# ── Output ────────────────────────────────────────────────────────────────────
save_recon_panel(params, panel_gt_pil, panel_idlg_pil, panel_masked_pil,
                 save_dir, block_idx, dataset, mask_desc, timestamp_str,
                 methods='both') -> path_str
# PNG grid; rows adapt to methods ('idlg','masked','both').

save_recon_gif(results_list, save_dir, block_idx, dataset, mask_desc,
               timestamp_str, methods='both', fps=8) -> path_str
# Animated GIF; rows=experiments, cols=[Init|iDLG?|Masked?|GT].
# results_list: list of result dicts from run_single_experiment.
```

---

## Key functions — helper/Network.py

```python
get_model(network: str, channel=3, num_classes=10, input_size=(32,32), pretrained=False) -> nn.Module
# Handles ALL architectures: 'LeNet','LeNet_bigger','MediumCNN','BiggerCNN'
# and any torchvision backbone (resnet*, vgg*, wide_resnet*, densenet*).
# pretrained=True loads ImageNet DEFAULT weights for torchvision models.

weights_init(m)   # uniform(-0.5, 0.5) for Conv2d / Linear layers
```

---

## run_single_experiment — config keys

All consumed from the `config` dict passed by `iDLG_mask.py`:

```
channel, num_classes, shape_img    dataset geometry
lr, num_dummy, Iteration           optimizer/attack params
MASK_MODE, PREFIXES, PREFIX_LAYER_FRACS, GRADSIZE_{TOPK,TOPFRAC,METRIC}
NETWORK_NAME                       string forwarded to get_model()
METHODS                            'idlg' | 'masked' | 'both'
COMPUTE_JACOBIAN_RANK, JACOBIAN_MAX_ENTRIES, JACOBIAN_SELECT_MODE
TV_WEIGHT, OPTIMIZER, NUM_RESTARTS, MAX_ITERATION, HISTORY_SIZE
SAVE_GIF, FRAME_INTERVAL           GIF frame capture (off by default)
run_id                             seed offset (seed = run_id + idx_net + 1)
EarlyStop: {loss_tol, patience, min_rel_improve, explode_factor, warmup, max_nan}
```

Result dict keys (sent via `result_queue.put()`):

On **subprocess failure**, only these keys are present: `error` (str), `traceback` (str), `idx_net`, `device_id`. The main loop checks `result.get('error')` before accessing any other key.

On **success**, the full dict contains:

```
idx_net, device_id
gt_data                            np [N,C,H,W]
final_recon                        dict {method: np [N,C,H,W]}
last_psnr_idlg / last_psnr_masked
last_loss_iDLG / last_mse_iDLG    final-iteration values
last_loss_iDLG_masked / last_mse_iDLG_masked
best_psnr_idlg / best_psnr_masked  best-restart values
best_loss_iDLG / best_mse_iDLG
best_loss_iDLG_masked / best_mse_iDLG_masked
label_iDLG / label_iDLG_masked    inferred labels
jac_rank_iDLG / jac_shape_iDLG
jac_rank_iDLG_masked / jac_shape_iDLG_masked
gt_label, imidx_list
early_stop_reason                  dict {method: reason_str}
early_stop_iter                    dict {method: iter_int}
init_frames                        dict {method: np [N,C,H,W]}   (if SAVE_GIF)
recon_frames                       dict {method: list of {iter,dummy,loss,mse}}
```

---

## iDLG_mask.py — argument summary

```
--dataset         MNIST|cifar10|cifar100|lfw        default: cifar100
--network         LeNet|LeNet_bigger|MediumCNN|BiggerCNN|resnetXX|vggXX
                                                    default: resnet18
--num_exp         int                               default: 10
--run_id          int   seed offset                 default: 0
--methods         idlg|masked|both                  default: idlg
--lr              float                             default: 1
--iteration       int                               default: 1000
--optimizer       lbfgs|adam|adamw|signed_adam|signed_adamw   default: lbfgs
--num_restarts    int                               default: 3
--max_iteration   int   LBFGS inner iters           default: 20
--history_size    int   LBFGS history               default: 100
--tv_weight       float                             default: 0.0
--mask_mode       see Masking modes table           default: gradsize_topfrac
--prefixes        str   'conv1:1.0,layer1:0.5,fc:1.0'
--gradsize_topk   int                               default: 20
--gradsize_topfrac float                            default: 0.5
--gradsize_metric l2|mean_abs|sum_abs               default: l2
--compute_jacobian_rank  flag
--jacobian_max_entries   int                        default: 4000
--jacobian_select_mode   topk_abs|first|random      default: topk_abs
--save_gif        flag
--frame_interval  int   iters between GIF frames    default: 20
--gif_fps         int                               default: 8
```

---

## Parallelism model

`iDLG_mask.py` spawns one `mp.Process` per GPU (spawn method, file_system sharing).  
Each process calls `run_single_experiment(idx_net, device_id, dst, dataset_name, config, result_queue)`.  
Results arrive out-of-order via `mp.SimpleQueue`; `all_results_by_idx[idx_net]` reorders them for GIF assembly.  
GPU count drives pool size: `min(num_gpus, num_exp)` processes start, and each freed GPU immediately takes the next experiment.

---

## Optimizers

| Name | Behaviour |
|---|---|
| `lbfgs` | Quasi-Newton; `MAX_ITERATION` inner steps per outer iter |
| `adam` | Adam with StepLR (step=300, γ=0.5) |
| `adamw` | Same with weight decay 1e-5 |
| `signed_adam` / `signed_adamw` | Sign-gradient variants of the above |

Dummy data is in logit space; `sigmoid(dummy_data)` gives image in [0,1].

---

## Early stopping signals

`plateau` — no relative improvement > `min_rel_improve` for `patience` steps after `warmup`  
`loss_tol` — loss < `loss_tol` after warmup  
`explosion` — current_loss > `explode_factor` × best_loss after warmup  
`nan_or_inf` — NaN/Inf loss (up to `max_nan` occurrences before break)  
`label_inference_unavailable` — final FC layer masked out  
`too_few_gradients` — observed entries < image pixel count  

---

## Output files

```
results/{timestamp}_{jobid}_results.csv        aggregate metrics per method per run
results/{timestamp}_{jobid}_{block}.png        reconstruction panel (PNG, 250 dpi)
results/{timestamp}_{jobid}_{block}_anim.gif   animated reconstruction GIF
```

CSV columns: `method, timestamp, job_id, dataset, network, restarts, lr, iteration,
num_exp, tv_weight, optimizer, max_iter, history, mask_mode, prefixes, grad_param,
med_best_loss, avg_best_loss, med_best_mse, avg_best_mse, avg_best_psnr, std_best_psnr, png_path`

---

## Gotchas

- **Baseline registry is per-run-id, single-run.** `results/baselines/idlg_baselines_registry.json` stores one entry per `(hyperparams, run_id)` hash. Running `--methods idlg` or `--methods both` with a given `run_id` saves/overwrites the baseline for that slot. `--methods masked` with the same `run_id` loads it automatically. The registry uses keys `best_psnr_list` / `best_mse_list` — old files with `avg_best_psnr_list` must be deleted and re-run.
- **Label inference gating** — if the final FC layer is masked, the experiment is skipped for that method (no gradient inversion possible without knowing the label).
- **`iters` scoping** — after early stop `break`, `iters` holds the break iteration. Final GIF frame is captured there if `_last_gif_iter != iters`.
- **Jacobian OOM** — `jacobian_max_entries` caps the row count; rows are built one at a time.
- **`gradsize_topk_entries` vs `gradsize_topk`** — former keeps top-K *scalar* entries; latter keeps top-K *tensors* (whole layers).
- **Normalization** — GT is normalized via `(x - dm) / ds` before `net()` and before Jacobian; dummy is `sigmoid(dummy_data)` then normalized the same way.
- **`weights_init` application** — `weights_init` (uniform init) is applied only to custom CNN architectures (`LeNet`, `LeNet_bigger`, `MediumCNN`, `BiggerCNN`) and only when `pretrained=False`. Torchvision backbones (ResNet, VGG, etc.) rely on PyTorch's own default initialization or their pretrained weights — do not apply `weights_init` to them.
- **`prefix_topfrac_entries_layer`** — passes `prefix_top_fracs` (dict) to `get_entry_masks_by_prefix_group`; each prefix can have its own retention fraction via `--prefixes conv1:1.0,layer1:0.5,...`.
