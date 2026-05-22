# Code Improvement Plan

**Project:** stable-ginv  
**Date:** 2026-05-22  
**Reviewed by:** Claude Code (claude-sonnet-4-6)

This plan is organized into two phases. Phase 1 contains safe, targeted improvements that can be done incrementally without breaking experiments. Phase 2 is a deeper structural refactor that should be done on a branch with careful testing.

---

## Critical Rules Before Touching Anything

1. **Never change gradient masking logic** without re-running a baseline experiment and comparing PSNR/SSIM to pre-change results.
2. **Never rename CSV column names** — downstream analysis depends on them.
3. **Never modify `invertinggradients/`** from this repo — it is a git submodule.
4. **Test after every change** using the smoke test in `docs/handover/HANDOVER_REVIEWER.md § How to Verify`.

---

## Phase 1 — Quick Wins

Estimated effort: 1–2 days. No architecture changes. Low breakage risk.

### P1.1 Split `Misc_functions.py` into focused modules ✅ DONE

**Problem:** `Misc_functions.py` is 1,170 lines and mixes: gradient masking, metrics, Jacobian rank, network construction, LR scheduling, visualization, and CSV I/O. It is the hardest file to navigate and contributes to confusion about what belongs where.

**Action:** Split into five files. Keep `Misc_functions.py` as a temporary re-export shim until all import sites are updated.

| New file | Functions to move |
|----------|-------------------|
| `functions/masking.py` | `flatten_observed_gradients`, `build_gradient_mask`, `get_keep_ids`, `get_keep_ids_by_gradsize`, `get_entry_masks_by_gradsize`, `get_prefix_keep_ids`, `get_entry_masks_by_prefix_group`, `get_keep_ids_by_prefix_group` |
| `helper/metrics.py` | `compute_psnr_from_mse`, `compute_ssim_batch`, `total_variation`, `compute_jacobian_rank`, `compute_grad_match_loss` |
| `helper/visualization.py` | `save_recon_panel`, `save_recon_gif` |
| `functions/io_utils.py` | `save_baseline_registry`, `load_baseline_registry`, `paired_summary`, CSV helpers |
| `helper/training_utils.py` | `build_network`, `make_scheduler` |

**Migration steps:**
1. Create each new file and move the functions (do not rename yet).
2. At the top of `Misc_functions.py`, replace the function bodies with re-exports:
   ```python
   from masking import flatten_observed_gradients, build_gradient_mask  # etc.
   ```
3. Verify smoke test passes.
4. Update import lines in all entry points (`iDLG_mask.py`, `run_single_exp.py`, `run_single_exp_batch.py`, `jacobian_parallel.py`, `jacobian_rank_sweep.py`) to import from the new modules directly.
5. Delete `Misc_functions.py` once all imports are updated.

---

### P1.2 Deduplicate normalization constants ✅ DONE

**Problem:** Dataset normalization constants (mean/std for CIFAR-10, CIFAR-100, MNIST, ImageNet, LFW) appear in both `consts.py` (root) and inline in `Misc_functions.py`. If a new dataset is added or LFW stats are recomputed, both must be updated — and this has already diverged once.

**Action:**
1. Audit `Misc_functions.py` for any inline definitions of `cifar10_mean`, `cifar100_mean`, `mnist_mean`, `lfw_mean`, `imagenet_mean`, and their `_std` counterparts.
2. Remove the inline definitions from `Misc_functions.py`.
3. Add `from consts import *` (or explicit imports) at the top of `Misc_functions.py` to get them from the single source of truth.
4. Confirm `consts.py` has all datasets including LFW. If LFW is only in `Misc_functions.py`, move it to `consts.py` first.

Do **not** touch `invertinggradients/inversefed/consts.py`.

---

### P1.3 Remove commented-out code

**Problem:** Several blocks of commented-out code add noise without adding value.

**Specific locations:**
- ~~`run_single_exp.py:10`~~ — done
- ~~`run_single_exp.py:68–70`~~ — done
- `iDLG_mask.py` — 5 commented-out `print` statements remain (lines 413, 436, 437, 550, 551, 586)
- ~~`run_single_exp_batch.py`~~ — archived

**Action:** Delete these lines entirely. If you think you might need them, they are in git history.

---

### P1.4 Add one-line docstrings to all public functions in the new modules ✅ DONE

**Problem:** No functions have docstrings. Parameters like `keep_ids`, `entry_masks`, `grad_list` are not self-explanatory from their names alone.

**Action:** After completing P1.1, add a one-line docstring to each function in the five new modules. The docstring should describe *what* the function returns and the key non-obvious parameter.

Example:
```python
def build_gradient_mask(named_params, grad_list, mask_mode, ...):
    """Return (keep_ids, entry_masks) for the given masking strategy; exactly one will be None."""
```

Do not write paragraph docstrings. One line per function. Focus on the return type and any non-obvious parameter contract.

---

### P1.5 Delete or archive `original/` directory ✅ DONE

**Problem:** `original/` contains a snapshot of an earlier version of the code. Nothing imports from it. It exists only for historical reference and confuses newcomers about what the "real" code is.

**Action:** Confirm nothing imports from `original/`. Then either:
- Tag the current commit as `pre-refactor` for historical reference, and delete `original/`.
- Or move it to a `archive/` folder if you want it visible but clearly marked as inactive.

---

### P1.6 Move `weights_init` to `Network.py` ✅ DONE

**Problem:** `weights_init` is defined in `Network.py` but also used in entry point scripts. Some entry points import it from `Network.py` directly while others get it via `Misc_functions`. This creates confusion.

**Action:** Confirm `weights_init` lives in `Network.py`. Remove any duplicate or re-export of it from `Misc_functions.py`. Update imports in entry points to use `from Network import weights_init`.

---

## Phase 2 — Structural Refactoring

Estimated effort: 3–5 days. Do this on a dedicated branch. Requires smoke-testing every entry point after completion.

### P2.1 Config dataclasses

**Problem:** All experiments are configured via a plain `dict` with string keys and no type validation. Typos in key names (e.g., `'Iterration'`) fail silently, using whatever default the `.get()` fallback provides.

**Action:** Replace the config dict with a `dataclass` (or `@dataclass` with `field` defaults):

```python
from dataclasses import dataclass, field
from typing import Optional, Tuple

@dataclass
class ExperimentConfig:
    network_name: str = 'resnet18'
    network_trained: bool = True
    channel: int = 3
    num_classes: int = 10
    shape_img: Tuple[int, int] = (32, 32)
    lr: float = 0.1
    gamma: float = 0.9
    num_dummy: int = 1
    num_iterations: int = 300       # replaces 'Iteration'
    mask_mode: str = 'none'
    gradsize_topk: int = 50
    gradsize_topfrac: float = 0.5
    gradsize_threshold: Optional[float] = None
    gradsize_metric: str = 'l2'
    prefixes: Tuple[str, ...] = field(default_factory=tuple)
    prefix_layer_fracs: dict = field(default_factory=dict)
    optimizer: str = 'lbfgs'
    max_lbfgs_iter: int = 20
    num_restarts: int = 1
    tv_weight: float = 0.0
    grad_loss: str = 'cos'
    compute_jacobian_rank: bool = False
    jacobian_max_entries: int = 4000
    save_gif: bool = False
    frame_interval: int = 20
    run_id: int = 0
```

**Note:** `'Iteration'` becomes `num_iterations` in the dataclass. Update all usages in worker functions at the same time — do not leave a mix of old and new keys.

---

### P2.2 Shared experiment core function

**Problem:** `iDLG_mask.py` and `run_single_exp.py` share 200–300 lines of nearly identical logic (network setup, dataset loading, gradient computation, masking, optimization loop). A bug fix in one does not automatically fix the other.

**Action:** Extract the shared logic into `experiment.py::run_experiment(config: ExperimentConfig, device: str) -> dict`. Both `iDLG_mask.py` and `run_single_exp.py` become thin wrappers that call this function.

The returned dict should contain: `psnr`, `ssim`, `loss`, `mse`, `recon_image`, `gt_image`, and optionally `jacobian_rank`.

---

### P2.3 Type hints throughout

**Action:** After P2.1 and P2.2, add type hints to all function signatures in:
1. `masking.py`
2. `metrics.py`
3. `experiment.py`
4. `Network.py`
5. `Dataset.py`

Key types to define:
```python
GradList = List[Optional[torch.Tensor]]
KeepIds = Optional[Set[int]]
EntryMasks = Optional[List[Optional[torch.Tensor]]]
```

---

### P2.4 Unit tests for masking logic

**Problem:** There are no unit tests. The only validation is manual smoke testing. Refactoring is risky because there is no automated way to confirm that masking produces the correct gradient selection.

**Action:** Create `tests/test_masking.py` with pytest tests for the core masking functions:

```python
# tests/test_masking.py
import torch
import pytest
from masking import build_gradient_mask, flatten_observed_gradients

def make_grad_list(shapes):
    return [torch.randn(s) for s in shapes]

def test_topk_keeps_k_tensors():
    grads = make_grad_list([(10,), (20,), (5,), (15,)])
    named = [('a', torch.empty(10)), ('b', torch.empty(20)),
             ('c', torch.empty(5)), ('d', torch.empty(15))]
    keep_ids, entry_masks = build_gradient_mask(named, grads, 'gradsize_topk',
                                                 gradsize_topk=2, gradsize_metric='l2')
    assert entry_masks is None
    assert len(keep_ids) == 2

def test_topfrac_fraction():
    grads = make_grad_list([(10,)] * 10)
    named = [(f'p{i}', torch.empty(10)) for i in range(10)]
    keep_ids, _ = build_gradient_mask(named, grads, 'gradsize_topfrac',
                                       gradsize_topfrac=0.3, gradsize_metric='l2')
    assert len(keep_ids) == 3  # 30% of 10

def test_none_mode_keeps_all():
    grads = make_grad_list([(10,), (20,)])
    named = [('a', torch.empty(10)), ('b', torch.empty(20))]
    keep_ids, entry_masks = build_gradient_mask(named, grads, 'none')
    assert keep_ids is None
    assert entry_masks is None

def test_flatten_with_keep_ids():
    grads = [torch.ones(5), torch.ones(3), torch.ones(4)]
    flat = flatten_observed_gradients(grads, keep_ids={0, 2})
    assert flat.shape == (9,)  # 5 + 4
```

Run with: `pytest tests/ -v`

---

### P2.5 Standardize entry point CLI (optional, lower priority)

**Problem:** Experiment configuration is only possible by editing Python source files. This makes it impossible to run different configurations without editing code, and makes cluster job scripts brittle.

**Action (optional):** Add `argparse` support to `iDLG_mask.py` and `run_single_exp.py` as an alternative to editing the config dict. The dict-based approach should remain as the default for backwards compatibility.

Example additions:
```bash
python run_single_exp.py --network resnet18 --dataset cifar10 --mask-mode gradsize_topfrac --topfrac 0.3 --iterations 300
```

This is lower priority than P2.1–P2.4. Only implement if running many different configs becomes a bottleneck.

---

## Summary Table

| Item | Phase | Effort | Risk | Impact |
|------|-------|--------|------|--------|
| P1.1 Split Misc_functions.py | 1 | Medium | Low | High | ✅ |
| P1.2 Deduplicate constants | 1 | Small | Low | Medium | ✅ |
| P1.3 Remove commented code | 1 | Small | Very Low | Low | 🔲 iDLG_mask.py remaining |
| P1.4 Add docstrings | 1 | Small | Very Low | Medium | ✅ |
| P1.5 Delete original/ | 1 | Small | Very Low | Low | ✅ |
| P1.6 Move weights_init | 1 | Small | Very Low | Low | ✅ |
| P2.1 Config dataclasses | 2 | Medium | Medium | High |
| P2.2 Shared experiment core | 2 | Large | Medium | High |
| P2.3 Type hints | 2 | Medium | Low | Medium |
| P2.4 Unit tests | 2 | Large | Low | High |
| P2.5 CLI argparse | 2 | Medium | Low | Low |

**Recommended order:** P1.1 → P1.2 → P1.3 → P1.4 → P1.5/P1.6 (can be done in any order after P1.1). Then P2.4 (write tests first so you have a safety net), then P2.1, P2.2, P2.3.
