# Phase 3 — `stable_ginv/masking/` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract `functions/masking.py` into `stable_ginv/masking/` with a `MaskStrategy` protocol, one strategy class per mask mode, a `STRATEGY_REGISTRY`, and a `Masker` facade — with no change to outputs, CLI, or golden hashes.

**Architecture:** All low-level computation functions move to `_helpers.py` and `_compute.py`; the 15 mask-mode branches of `build_gradient_mask` each become a strategy class in `strategies.py`; the `Masker` facade in `facade.py` dispatches via `STRATEGY_REGISTRY`; `build_gradient_mask` becomes a one-liner wrapper over `Masker`. The old `functions/masking.py` becomes a thin re-export shim so all existing callers (`run_single_exp.py`, `functions/jacobian_rank_sweep.py`, `functions/rank_reconstruction_plot.py`) keep working without changes.

**Tech Stack:** Python 3, PyTorch, `typing.Protocol`, pytest. Conda env: `stable-ginv`. All verification via `conda run -n stable-ginv`.

---

## Invariants (enforce every task)

- `conda run -n stable-ginv python -m pytest tests/ -q` stays green throughout.
- `conda run -n stable-ginv python iDLG_mask.py --help` works throughout.
- No change to reconstruction math, CLI defaults, CSV columns, or golden hashes.
- `functions/masking.py` is a shim by end of phase — callers that still import from it continue to work.

## File map

| Action | Path | Purpose |
|--------|------|---------|
| Create | `stable_ginv/masking/__init__.py` | Public API: `Masker`, `build_gradient_mask`, `flatten_observed_gradients`, `_get_last_fc_param_indices`, `MaskStrategy`, `STRATEGY_REGISTRY` |
| Create | `stable_ginv/masking/_helpers.py` | Private helpers: `_get_last_fc_param_indices`, `_is_vgg`, `_gradsize_entries_layer_param_names`, `_grad_magnitude` |
| Create | `stable_ginv/masking/_compute.py` | Low-level functions: `flatten_observed_gradients`, `get_keep_ids_by_gradsize`, `get_entry_masks_by_gradsize`, `get_entry_masks_by_prefix_group`, `get_keep_ids_by_prefix_group`, `get_prefix_keep_ids`, `get_keep_ids` |
| Create | `stable_ginv/masking/strategies.py` | `MaskStrategy` protocol, one class per mode, `STRATEGY_REGISTRY` |
| Create | `stable_ginv/masking/facade.py` | `Masker` class + `build_gradient_mask` wrapper |
| Create | `tests/test_masker.py` | Tests for `Masker` and `STRATEGY_REGISTRY` (new interface) |
| Modify | `functions/masking.py` | Replace with re-export shim |
| Modify | `stable_ginv/metrics/jacobian.py` | Update import: `functions.masking` → `stable_ginv.masking` |
| Modify | `tests/test_masking.py` | Update import: `functions.masking` → `stable_ginv.masking` |
| Modify | `tests/golden/test_masking_golden.py` | Update import: `functions.masking` → `stable_ginv.masking` |
| Modify | `docs/handover/HANDOVER_RESTRUCTURE.md` | Mark Phase 3 complete, update Next phase |

---

## Task 1: Create package skeleton and write failing test

**Files:**
- Create: `stable_ginv/masking/__init__.py` (empty skeleton)
- Create: `tests/test_masker.py`

- [ ] **Step 1: Create the empty package directory and skeleton `__init__.py`**

```bash
mkdir -p /home/mathias/GitHub/stable-ginv/stable_ginv/masking
```

Create `stable_ginv/masking/__init__.py` with content:
```python
# Populated in Task 5.
```

- [ ] **Step 2: Write `tests/test_masker.py`**

```python
"""Tests for the Masker facade and STRATEGY_REGISTRY (Phase 3)."""
import torch
import torch.nn as nn

from stable_ginv.masking import Masker, STRATEGY_REGISTRY, build_gradient_mask


def _small_net():
    return nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))


def _small_grads(net):
    return [torch.ones_like(p) for p in net.parameters()]


def test_masker_gradsize_topk_matches_build_gradient_mask():
    net = _small_net()
    grads = [torch.full((1,), float(v)) for v in [1, 5, 2, 10]]
    expected_ids, _ = build_gradient_mask("masked", "gradsize_topk", net, grads, gradsize_topk=2)
    ids, masks = Masker("masked", "gradsize_topk", gradsize_topk=2).apply(net, grads)
    assert ids == expected_ids
    assert masks is None


def test_masker_idlg_keeps_all_params():
    net = _small_net()
    grads = _small_grads(net)
    ids, masks = Masker("idlg", "gradsize_topk").apply(net, grads)
    assert ids == set(range(4))
    assert masks is None


def test_masker_entry_mode_returns_none_keep_ids():
    net = _small_net()
    grads = _small_grads(net)
    ids, masks = Masker("masked", "gradsize_topfrac_entries", gradsize_topfrac=0.5).apply(net, grads)
    assert ids is None
    assert masks is not None
    assert len(masks) == 4


def test_masker_prefix_mode():
    net = _small_net()
    grads = _small_grads(net)
    ids, masks = Masker("masked", "prefix", prefixes=("0",)).apply(net, grads)
    assert ids == {0, 1}
    assert masks is None


def test_strategy_registry_contains_all_modes():
    expected = {
        "gradsize_topk", "gradsize_topfrac",
        "gradsize_topk_entries", "gradsize_topfrac_entries",
        "gradsize_topk_entries_layer", "gradsize_topfrac_entries_layer",
        "prefix", "prefix_topk", "prefix_topfrac",
        "prefix_topk_entries", "prefix_topfrac_entries",
        "prefix_topk_entries_layer", "prefix_topfrac_entries_layer",
    }
    assert expected.issubset(set(STRATEGY_REGISTRY.keys()))
```

- [ ] **Step 3: Run the new tests and confirm they fail with ImportError**

```bash
conda run -n stable-ginv python -m pytest tests/test_masker.py -v
```

Expected: FAIL — `ImportError: cannot import name 'Masker' from 'stable_ginv.masking'`

---

## Task 2: Implement `stable_ginv/masking/_helpers.py`

**Files:**
- Create: `stable_ginv/masking/_helpers.py`

- [ ] **Step 1: Create `stable_ginv/masking/_helpers.py`**

```python
import torch.nn as nn


def _get_last_fc_param_indices(net):
    """Return parameter indices (weight + bias) of the last nn.Linear in the network."""
    last_linear_name = None
    for name, module in net.named_modules():
        if isinstance(module, nn.Linear):
            last_linear_name = name
    if last_linear_name is None:
        return set()
    indices = set()
    for idx, (pname, _) in enumerate(net.named_parameters()):
        if pname in (f"{last_linear_name}.weight", f"{last_linear_name}.bias"):
            indices.add(idx)
    return indices


def _is_vgg(net):
    return net.__class__.__name__.lower() == "vgg"


def _gradsize_entries_layer_param_names(net):
    """Parameter groups for per-layer entry masking; VGG classifier heads excluded."""
    named_params = list(net.named_parameters())
    if not _is_vgg(net):
        return tuple(name for name, _ in named_params)
    # VGG intermediate classifier layers dominate parameter count and slow
    # create_graph=True reconstruction. Exclude classifier.*; label inference
    # uses original final-layer gradients before masking.
    return tuple(name for name, _ in named_params if not name.startswith("classifier."))


def _grad_magnitude(g, metric):
    """Scalar gradient magnitude for a single tensor using the given metric."""
    if metric == "l2":       return g.detach().norm(p=2).item()
    if metric == "mean_abs": return g.detach().abs().mean().item()
    if metric == "sum_abs":  return g.detach().abs().sum().item()
    raise ValueError(f"Unknown metric: {metric}")
```

---

## Task 3: Implement `stable_ginv/masking/_compute.py`

**Files:**
- Create: `stable_ginv/masking/_compute.py`

- [ ] **Step 1: Create `stable_ginv/masking/_compute.py`**

```python
import torch

from stable_ginv.masking._helpers import _grad_magnitude


def flatten_observed_gradients(grad_list, keep_ids=None, entry_masks=None):
    """Flatten selected gradient entries; keep_ids and entry_masks are mutually exclusive."""
    flat = []
    for i, g in enumerate(grad_list):
        if g is None:
            continue
        if entry_masks is not None:
            m = entry_masks[i]
            if m is None:
                continue
            flat.append(g.reshape(-1)[m.reshape(-1)])
        else:
            if keep_ids is not None and i not in keep_ids:
                continue
            flat.append(g.reshape(-1))
    if len(flat) == 0:
        raise ValueError("No observed gradients selected.")
    return torch.cat(flat, dim=0)


def get_keep_ids_by_gradsize(
    original_dy_dx, mode="topk", topk=10, top_frac=None, metric="l2", candidate_ids=None
):
    """Tensor-wise masking: return (sorted keep_ids, sizes_sorted) by gradient magnitude."""
    sizes = []
    for i, g in enumerate(original_dy_dx):
        if candidate_ids is not None and i not in candidate_ids:
            continue
        if g is None:
            sizes.append((i, float("-inf")))
            continue
        sizes.append((i, _grad_magnitude(g, metric)))

    sizes_sorted = sorted(sizes, key=lambda x: x[1], reverse=True)

    if mode == "topk":
        k = min(topk, len(sizes_sorted))
        keep = [i for i, _ in sizes_sorted[:k]]
    elif mode == "topfrac":
        if top_frac is None:
            raise ValueError("top_frac must be set for mode='topfrac'")
        k = max(1, int(round(top_frac * len(sizes_sorted))))
        keep = [i for i, _ in sizes_sorted[:k]]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return sorted(keep), sizes_sorted


def get_entry_masks_by_gradsize(
    original_dy_dx, mode="topk_entries", topk=None, top_frac=None, candidate_ids=None
):
    """Entry-wise masking by global gradient magnitude; returns (entry_masks, kept, total)."""
    pieces = []
    meta = []

    for i, g in enumerate(original_dy_dx):
        if candidate_ids is not None and i not in candidate_ids:
            meta.append((i, 0, None))
            continue
        if g is None:
            meta.append((i, 0, None))
            continue
        flat = g.detach().abs().reshape(-1)
        pieces.append(flat)
        meta.append((i, flat.numel(), g.shape))

    if len(pieces) == 0:
        raise ValueError("No gradients available for masking.")

    all_vals = torch.cat(pieces, dim=0)
    total_entries = all_vals.numel()

    if mode == "topk_entries":
        if topk is None:
            raise ValueError("topk must be set for mode='topk_entries'")
        k = max(1, min(int(topk), total_entries))
    elif mode == "topfrac_entries":
        if top_frac is None:
            raise ValueError("top_frac must be set for mode='topfrac_entries'")
        k = max(1, min(int(round(top_frac * total_entries)), total_entries))
    else:
        raise ValueError(f"Unknown entrywise mode: {mode}")

    top_idx = torch.topk(all_vals, k=k, largest=True).indices
    global_mask = torch.zeros(total_entries, dtype=torch.bool, device=all_vals.device)
    global_mask[top_idx] = True

    entry_masks = []
    offset = 0
    for i, numel, shape in meta:
        g = original_dy_dx[i]
        if g is None or shape is None or numel == 0:
            entry_masks.append(None)
            continue
        local_mask = global_mask[offset:offset + numel].reshape(shape)
        entry_masks.append(local_mask)
        offset += numel

    return entry_masks, k, total_entries


def get_entry_masks_by_prefix_group(
    net,
    original_dy_dx,
    prefixes,
    mode="topfrac_entries",
    topk=None,
    top_frac=None,
    prefix_top_fracs=None,
):
    """Entry-wise masking ranked within each prefix group; returns (entry_masks, kept, total)."""
    named_params = list(net.named_parameters())
    entry_masks = [None] * len(original_dy_dx)

    kept_entries = 0
    total_entries = 0

    if prefix_top_fracs is None:
        prefix_top_fracs = {}

    for prefix in prefixes:
        group_infos = []
        pieces = []

        for i, (name, _) in enumerate(named_params):
            if not (name == prefix or name.startswith(prefix + ".")):
                continue
            g = original_dy_dx[i]
            if g is None:
                continue
            flat = g.detach().abs().reshape(-1)
            group_infos.append((i, g.shape, flat.numel()))
            pieces.append(flat)

        if len(pieces) == 0:
            continue

        all_vals = torch.cat(pieces, dim=0)
        n = all_vals.numel()
        total_entries += n

        if mode == "topk_entries":
            if topk is None:
                raise ValueError("topk must be set for mode='topk_entries'")
            k = max(1, min(int(topk), n))
        elif mode == "topfrac_entries":
            local_top_frac = prefix_top_fracs.get(prefix, top_frac)
            if local_top_frac is None:
                raise ValueError("top_frac must be set for mode='topfrac_entries'")
            if local_top_frac == 0:
                continue
            k = max(1, min(int(round(local_top_frac * n)), n))
        else:
            raise ValueError(f"Unknown mode: {mode}")

        top_idx = torch.topk(all_vals, k=k, largest=True).indices
        group_mask = torch.zeros(n, dtype=torch.bool, device=all_vals.device)
        group_mask[top_idx] = True

        offset = 0
        for i, shape, numel in group_infos:
            local_mask = group_mask[offset:offset + numel].reshape(shape)
            entry_masks[i] = local_mask
            offset += numel

        kept_entries += k

    if kept_entries == 0:
        raise ValueError(f"No gradients matched prefixes={prefixes}")

    return entry_masks, kept_entries, total_entries


def get_keep_ids_by_prefix_group(
    net,
    original_dy_dx,
    prefixes,
    mode="topfrac",
    topk=None,
    top_frac=None,
    prefix_top_fracs=None,
    prefix_top_ks=None,
    metric="l2",
):
    """Tensor-wise masking ranked within each prefix group; returns (sorted keep_ids, ranked_by_prefix)."""
    if prefix_top_fracs is None:
        prefix_top_fracs = {}
    if prefix_top_ks is None:
        prefix_top_ks = {}

    named_params = list(net.named_parameters())
    keep = set()
    ranked_by_prefix = {}

    for prefix in prefixes:
        sizes = []
        for i, (name, _) in enumerate(named_params):
            if not (name == prefix or name.startswith(prefix + ".")):
                continue
            g = original_dy_dx[i]
            if g is None:
                continue
            sizes.append((i, _grad_magnitude(g, metric)))

        if len(sizes) == 0:
            continue

        sizes_sorted = sorted(sizes, key=lambda x: x[1], reverse=True)
        ranked_by_prefix[prefix] = sizes_sorted

        if mode == "topk":
            local_topk = prefix_top_ks.get(prefix, topk)
            if local_topk is None:
                raise ValueError("topk must be set for mode='topk'")
            k = max(0, min(int(local_topk), len(sizes_sorted)))
        elif mode == "topfrac":
            local_top_frac = prefix_top_fracs.get(prefix, top_frac)
            if local_top_frac is None:
                raise ValueError("top_frac must be set for mode='topfrac'")
            if local_top_frac == 0:
                k = 0
            else:
                k = max(1, min(int(round(local_top_frac * len(sizes_sorted))), len(sizes_sorted)))
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if k > 0:
            keep.update(i for i, _ in sizes_sorted[:k])

    if len(keep) == 0:
        raise ValueError(f"No parameters matched prefixes={prefixes}")

    return sorted(keep), ranked_by_prefix


def get_prefix_keep_ids(net, prefixes):
    """Return set of parameter indices whose name starts with any of the given prefixes."""
    keep = set()
    for idx, (name, _) in enumerate(net.named_parameters()):
        if any(name == p or name.startswith(p + ".") for p in prefixes):
            keep.add(idx)
    if len(keep) == 0:
        raise ValueError(f"No parameters matched prefixes={prefixes}")
    return keep


def get_keep_ids(mask_mode, net=None, prefixes=None):
    """Return set of parameter indices to keep for the given mask_mode."""
    if prefixes is not None:
        if net is None:
            raise ValueError("prefix-based keep_ids requires 'net'")
        return get_prefix_keep_ids(net, prefixes)

    if mask_mode == "all":
        if net is None:
            return set(range(8))
        return set(range(len(list(net.parameters()))))

    raise ValueError(f"Unknown MASK_MODE: {mask_mode}")
```

---

## Task 4: Implement `stable_ginv/masking/strategies.py`

**Files:**
- Create: `stable_ginv/masking/strategies.py`

Context: Each class below encapsulates exactly one branch of the original `build_gradient_mask` if/elif chain. The `method == "idlg"` case becomes `_IdlgStrategy` (used directly in `Masker`, not registered). The `else` fallback (handles `"all"` mode) is handled directly in `Masker.apply`, not via the registry.

Note the asymmetry between `gradsize_topk_entries_layer` (uses ALL param names, no VGG exclusion) and `gradsize_topfrac_entries_layer` (uses `_gradsize_entries_layer_param_names`, excludes VGG classifier). This matches the original exactly.

- [ ] **Step 1: Create `stable_ginv/masking/strategies.py`**

```python
from typing import Protocol

from stable_ginv.masking._helpers import _gradsize_entries_layer_param_names
from stable_ginv.masking._compute import (
    get_keep_ids,
    get_keep_ids_by_gradsize,
    get_entry_masks_by_gradsize,
    get_entry_masks_by_prefix_group,
    get_keep_ids_by_prefix_group,
    get_prefix_keep_ids,
)


class MaskStrategy(Protocol):
    def apply(
        self,
        net,
        original_dy_dx,
        prefixes,
        prefix_layer_fracs,
        gradsize_topk,
        gradsize_topfrac,
        gradsize_metric,
    ) -> tuple: ...


class _IdlgStrategy:
    def apply(self, net, original_dy_dx, prefixes, prefix_layer_fracs,
              gradsize_topk, gradsize_topfrac, gradsize_metric):
        return get_keep_ids("all", net=net), None


class _GradsizeTopKStrategy:
    def apply(self, net, original_dy_dx, prefixes, prefix_layer_fracs,
              gradsize_topk, gradsize_topfrac, gradsize_metric):
        keep_ids, _ = get_keep_ids_by_gradsize(
            original_dy_dx, mode="topk", topk=gradsize_topk, metric=gradsize_metric
        )
        return keep_ids, None


class _GradsizeTopFracStrategy:
    def apply(self, net, original_dy_dx, prefixes, prefix_layer_fracs,
              gradsize_topk, gradsize_topfrac, gradsize_metric):
        keep_ids, _ = get_keep_ids_by_gradsize(
            original_dy_dx, mode="topfrac", top_frac=gradsize_topfrac, metric=gradsize_metric
        )
        return keep_ids, None


class _GradsizeTopKEntriesStrategy:
    def apply(self, net, original_dy_dx, prefixes, prefix_layer_fracs,
              gradsize_topk, gradsize_topfrac, gradsize_metric):
        entry_masks, _, _ = get_entry_masks_by_gradsize(
            original_dy_dx, mode="topk_entries", topk=gradsize_topk
        )
        return None, entry_masks


class _GradsizeTopFracEntriesStrategy:
    def apply(self, net, original_dy_dx, prefixes, prefix_layer_fracs,
              gradsize_topk, gradsize_topfrac, gradsize_metric):
        entry_masks, _, _ = get_entry_masks_by_gradsize(
            original_dy_dx, mode="topfrac_entries", top_frac=gradsize_topfrac
        )
        return None, entry_masks


class _GradsizeTopKEntriesLayerStrategy:
    def apply(self, net, original_dy_dx, prefixes, prefix_layer_fracs,
              gradsize_topk, gradsize_topfrac, gradsize_metric):
        # All param names — no VGG exclusion (matches original build_gradient_mask).
        all_param_names = tuple(name for name, _ in net.named_parameters())
        entry_masks, _, _ = get_entry_masks_by_prefix_group(
            net=net, original_dy_dx=original_dy_dx, prefixes=all_param_names,
            mode="topk_entries", topk=gradsize_topk,
        )
        return None, entry_masks


class _GradsizeTopFracEntriesLayerStrategy:
    def apply(self, net, original_dy_dx, prefixes, prefix_layer_fracs,
              gradsize_topk, gradsize_topfrac, gradsize_metric):
        # VGG classifier excluded via _gradsize_entries_layer_param_names.
        all_param_names = _gradsize_entries_layer_param_names(net)
        entry_masks, _, _ = get_entry_masks_by_prefix_group(
            net=net, original_dy_dx=original_dy_dx, prefixes=all_param_names,
            mode="topfrac_entries", top_frac=gradsize_topfrac,
        )
        return None, entry_masks


class _PrefixTopKStrategy:
    def apply(self, net, original_dy_dx, prefixes, prefix_layer_fracs,
              gradsize_topk, gradsize_topfrac, gradsize_metric):
        keep_ids, _ = get_keep_ids_by_prefix_group(
            net=net, original_dy_dx=original_dy_dx, prefixes=prefixes,
            mode="topk", topk=gradsize_topk,
            prefix_top_ks={k: int(v) for k, v in prefix_layer_fracs.items()},
            metric=gradsize_metric,
        )
        return keep_ids, None


class _PrefixTopFracStrategy:
    def apply(self, net, original_dy_dx, prefixes, prefix_layer_fracs,
              gradsize_topk, gradsize_topfrac, gradsize_metric):
        keep_ids, _ = get_keep_ids_by_prefix_group(
            net=net, original_dy_dx=original_dy_dx, prefixes=prefixes,
            mode="topfrac", top_frac=gradsize_topfrac,
            prefix_top_fracs=prefix_layer_fracs,
            metric=gradsize_metric,
        )
        return keep_ids, None


class _PrefixTopKEntriesStrategy:
    def apply(self, net, original_dy_dx, prefixes, prefix_layer_fracs,
              gradsize_topk, gradsize_topfrac, gradsize_metric):
        candidate_ids = get_prefix_keep_ids(net, prefixes)
        entry_masks, _, _ = get_entry_masks_by_gradsize(
            original_dy_dx, mode="topk_entries", topk=gradsize_topk,
            candidate_ids=candidate_ids,
        )
        return None, entry_masks


class _PrefixTopFracEntriesStrategy:
    def apply(self, net, original_dy_dx, prefixes, prefix_layer_fracs,
              gradsize_topk, gradsize_topfrac, gradsize_metric):
        candidate_ids = get_prefix_keep_ids(net, prefixes)
        entry_masks, _, _ = get_entry_masks_by_gradsize(
            original_dy_dx, mode="topfrac_entries", top_frac=gradsize_topfrac,
            candidate_ids=candidate_ids,
        )
        return None, entry_masks


class _PrefixTopKEntriesLayerStrategy:
    def apply(self, net, original_dy_dx, prefixes, prefix_layer_fracs,
              gradsize_topk, gradsize_topfrac, gradsize_metric):
        entry_masks, _, _ = get_entry_masks_by_prefix_group(
            net=net, original_dy_dx=original_dy_dx, prefixes=prefixes,
            mode="topk_entries", topk=gradsize_topk,
            prefix_top_fracs=prefix_layer_fracs,
        )
        return None, entry_masks


class _PrefixTopFracEntriesLayerStrategy:
    def apply(self, net, original_dy_dx, prefixes, prefix_layer_fracs,
              gradsize_topk, gradsize_topfrac, gradsize_metric):
        entry_masks, _, _ = get_entry_masks_by_prefix_group(
            net=net, original_dy_dx=original_dy_dx, prefixes=prefixes,
            mode="topfrac_entries", top_frac=gradsize_topfrac,
            prefix_top_fracs=prefix_layer_fracs,
        )
        return None, entry_masks


class _PrefixStrategy:
    def apply(self, net, original_dy_dx, prefixes, prefix_layer_fracs,
              gradsize_topk, gradsize_topfrac, gradsize_metric):
        return get_keep_ids(mask_mode="prefix", net=net, prefixes=prefixes), None


STRATEGY_REGISTRY: dict = {
    "gradsize_topk":               _GradsizeTopKStrategy,
    "gradsize_topfrac":            _GradsizeTopFracStrategy,
    "gradsize_topk_entries":       _GradsizeTopKEntriesStrategy,
    "gradsize_topfrac_entries":    _GradsizeTopFracEntriesStrategy,
    "gradsize_topk_entries_layer": _GradsizeTopKEntriesLayerStrategy,
    "gradsize_topfrac_entries_layer": _GradsizeTopFracEntriesLayerStrategy,
    "prefix_topk":                 _PrefixTopKStrategy,
    "prefix_topfrac":              _PrefixTopFracStrategy,
    "prefix_topk_entries":         _PrefixTopKEntriesStrategy,
    "prefix_topfrac_entries":      _PrefixTopFracEntriesStrategy,
    "prefix_topk_entries_layer":   _PrefixTopKEntriesLayerStrategy,
    "prefix_topfrac_entries_layer": _PrefixTopFracEntriesLayerStrategy,
    "prefix":                      _PrefixStrategy,
}
```

---

## Task 5: Implement `facade.py` and fill `__init__.py`

**Files:**
- Create: `stable_ginv/masking/facade.py`
- Modify: `stable_ginv/masking/__init__.py`

- [ ] **Step 1: Create `stable_ginv/masking/facade.py`**

```python
from stable_ginv.masking.strategies import _IdlgStrategy, STRATEGY_REGISTRY
from stable_ginv.masking._compute import get_keep_ids


class Masker:
    """Selects and applies the right masking strategy from STRATEGY_REGISTRY."""

    def __init__(
        self,
        method: str,
        mask_mode: str,
        prefixes: tuple = (),
        prefix_layer_fracs: dict = None,
        gradsize_topk: int = 20,
        gradsize_topfrac: float = 0.5,
        gradsize_metric: str = "l2",
    ):
        self._method = method
        self._mask_mode = mask_mode
        self._prefixes = prefixes
        self._prefix_layer_fracs = prefix_layer_fracs or {}
        self._gradsize_topk = gradsize_topk
        self._gradsize_topfrac = gradsize_topfrac
        self._gradsize_metric = gradsize_metric

    def apply(self, net, original_dy_dx):
        if self._method == "idlg":
            strategy = _IdlgStrategy()
        elif self._mask_mode in STRATEGY_REGISTRY:
            strategy = STRATEGY_REGISTRY[self._mask_mode]()
        else:
            # Handles "all" and any unknown mode via get_keep_ids (raises for truly unknown).
            return get_keep_ids(self._mask_mode, net=net), None

        return strategy.apply(
            net,
            original_dy_dx,
            self._prefixes,
            self._prefix_layer_fracs,
            self._gradsize_topk,
            self._gradsize_topfrac,
            self._gradsize_metric,
        )


def build_gradient_mask(
    method,
    mask_mode,
    net,
    original_dy_dx,
    prefixes=(),
    prefix_layer_fracs=None,
    gradsize_topk=20,
    gradsize_topfrac=0.5,
    gradsize_metric="l2",
):
    """Dispatch to the appropriate masking strategy; returns (keep_ids, entry_masks), exactly one None."""
    return Masker(
        method, mask_mode, prefixes, prefix_layer_fracs,
        gradsize_topk, gradsize_topfrac, gradsize_metric,
    ).apply(net, original_dy_dx)
```

- [ ] **Step 2: Replace `stable_ginv/masking/__init__.py` with the full public API**

```python
from stable_ginv.masking._helpers import _get_last_fc_param_indices
from stable_ginv.masking._compute import flatten_observed_gradients
from stable_ginv.masking.strategies import MaskStrategy, STRATEGY_REGISTRY
from stable_ginv.masking.facade import Masker, build_gradient_mask

__all__ = [
    "Masker",
    "build_gradient_mask",
    "flatten_observed_gradients",
    "_get_last_fc_param_indices",
    "MaskStrategy",
    "STRATEGY_REGISTRY",
]
```

---

## Task 6: Verify new tests pass against `stable_ginv.masking`

**Files:** (none changed)

- [ ] **Step 1: Run only `test_masker.py`**

```bash
conda run -n stable-ginv python -m pytest tests/test_masker.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 2: Run the full suite to confirm nothing regressed**

```bash
conda run -n stable-ginv python -m pytest tests/ -q
```

Expected: all existing tests PASS (they still import from `functions.masking`, which is unchanged).

---

## Task 7: Convert `functions/masking.py` to a re-export shim

**Files:**
- Modify: `functions/masking.py`

The shim must export everything that existing callers import:
- `run_single_exp.py`: `build_gradient_mask`, `_get_last_fc_param_indices`
- `functions/jacobian_rank_sweep.py`: `build_gradient_mask`
- `functions/rank_reconstruction_plot.py`: `_get_last_fc_param_indices`, `flatten_observed_gradients`
- `tests/test_masking.py`: `flatten_observed_gradients`, `get_keep_ids_by_gradsize`, `get_entry_masks_by_gradsize`, `build_gradient_mask`
- `tests/golden/test_masking_golden.py`: `build_gradient_mask`

- [ ] **Step 1: Replace the entire content of `functions/masking.py`**

```python
"""Shim: stable_ginv.masking is the canonical home (Phase 3)."""
from stable_ginv.masking import (
    Masker,
    build_gradient_mask,
    flatten_observed_gradients,
    _get_last_fc_param_indices,
    MaskStrategy,
    STRATEGY_REGISTRY,
)
from stable_ginv.masking._helpers import (
    _is_vgg,
    _gradsize_entries_layer_param_names,
    _grad_magnitude,
)
from stable_ginv.masking._compute import (
    get_keep_ids_by_gradsize,
    get_entry_masks_by_gradsize,
    get_entry_masks_by_prefix_group,
    get_keep_ids_by_prefix_group,
    get_prefix_keep_ids,
    get_keep_ids,
)

__all__ = [
    "Masker",
    "build_gradient_mask",
    "flatten_observed_gradients",
    "_get_last_fc_param_indices",
    "MaskStrategy",
    "STRATEGY_REGISTRY",
    "_is_vgg",
    "_gradsize_entries_layer_param_names",
    "_grad_magnitude",
    "get_keep_ids_by_gradsize",
    "get_entry_masks_by_gradsize",
    "get_entry_masks_by_prefix_group",
    "get_keep_ids_by_prefix_group",
    "get_prefix_keep_ids",
    "get_keep_ids",
]
```

- [ ] **Step 2: Run the full test suite to confirm shim works**

```bash
conda run -n stable-ginv python -m pytest tests/ -q
```

Expected: all tests PASS (including goldens — they import `build_gradient_mask` from `functions.masking`, now served via shim).

- [ ] **Step 3: Confirm CLI still works**

```bash
conda run -n stable-ginv python iDLG_mask.py --help
```

Expected: help text printed, exit 0.

---

## Task 8: Update `stable_ginv/metrics/jacobian.py` to import from `stable_ginv.masking`

**Files:**
- Modify: `stable_ginv/metrics/jacobian.py`

`stable_ginv/metrics/jacobian.py` already lives inside the package, so it should import from `stable_ginv.masking` directly rather than from `functions.masking`.

- [ ] **Step 1: Find the import line**

```bash
grep -n "from functions.masking" stable_ginv/metrics/jacobian.py
```

Expected output (line number may vary):
```
5:from functions.masking import flatten_observed_gradients
```

- [ ] **Step 2: Replace that import line**

Change:
```python
from functions.masking import flatten_observed_gradients
```

To:
```python
from stable_ginv.masking import flatten_observed_gradients
```

- [ ] **Step 3: Run the full suite**

```bash
conda run -n stable-ginv python -m pytest tests/ -q
```

Expected: all tests PASS.

---

## Task 9: Update test imports to point at `stable_ginv.masking`

**Files:**
- Modify: `tests/test_masking.py`
- Modify: `tests/golden/test_masking_golden.py`

Tests should import from the new canonical location. The shim means existing imports work, but updating the tests makes the dependency explicit and means they no longer go through the shim.

- [ ] **Step 1: Update `tests/test_masking.py`**

Find the import block near the top of the file:
```python
from functions.masking import (
    flatten_observed_gradients,
    get_keep_ids_by_gradsize,
    get_entry_masks_by_gradsize,
    build_gradient_mask,
)
```

Replace with:
```python
from stable_ginv.masking import (
    flatten_observed_gradients,
    build_gradient_mask,
)
from stable_ginv.masking._compute import (
    get_keep_ids_by_gradsize,
    get_entry_masks_by_gradsize,
)
```

- [ ] **Step 2: Update `tests/golden/test_masking_golden.py`**

Find the import line near line 11:
```python
from functions.masking import build_gradient_mask
```

Replace with:
```python
from stable_ginv.masking import build_gradient_mask
```

- [ ] **Step 3: Run the full suite**

```bash
conda run -n stable-ginv python -m pytest tests/ -q
```

Expected: all tests PASS (golden hashes must be unchanged — if any differ, STOP and debug before proceeding).

---

## Task 10: Commit `refactor:` — new masking package + shim + jacobian update

- [ ] **Step 1: Stage the refactor files**

```bash
git add stable_ginv/masking/ functions/masking.py stable_ginv/metrics/jacobian.py
```

- [ ] **Step 2: Commit**

```bash
git commit -m "refactor: extract stable_ginv/masking/ strategy classes from functions/masking.py"
```

---

## Task 11: Commit `test:` — new Masker tests + updated test imports

- [ ] **Step 1: Stage the test files**

```bash
git add tests/test_masker.py tests/test_masking.py tests/golden/test_masking_golden.py
```

- [ ] **Step 2: Commit**

```bash
git commit -m "test: add Masker tests; update test imports to stable_ginv.masking"
```

---

## Task 12: Update `HANDOVER_RESTRUCTURE.md` and commit `docs:`

**Files:**
- Modify: `docs/handover/HANDOVER_RESTRUCTURE.md`

- [ ] **Step 1: In `HANDOVER_RESTRUCTURE.md`, update the Status section**

Find:
```markdown
- [ ] Phase 3 — `stable_ginv/masking/` strategy classes.
```

Replace with:
```markdown
- [x] Phase 3 — `stable_ginv/masking/` strategy classes.
```

- [ ] **Step 2: Update the "Next phase" section**

Find and replace the `## Next phase` section with:
```markdown
## Next phase

Write the Phase 4 plan from the spec, then implement. Phase 4 tears down
`functions/io_utils.py` (732 lines) into `stable_ginv/registry/`,
`stable_ginv/stats/`, and `stable_ginv/io/`. Add CSV-row golden tests just
before moving registry load/save logic. Keep this file's Status section
current at every phase boundary.
```

- [ ] **Step 3: Run a final full verification**

```bash
conda run -n stable-ginv python -m pytest tests/ -q
conda run -n stable-ginv python iDLG_mask.py --help
git diff --check
```

Expected: all PASS, no whitespace errors.

- [ ] **Step 4: Commit**

```bash
git add docs/handover/HANDOVER_RESTRUCTURE.md
git commit -m "docs: mark Phase 3 complete in HANDOVER_RESTRUCTURE"
```

---

## Self-review checklist

**Spec coverage:**
- [x] `MaskStrategy` protocol defined in `strategies.py`
- [x] One strategy class per mode (13 modes + idlg = 14 classes)
- [x] `STRATEGY_REGISTRY` keyed by mode name
- [x] `Masker` facade dispatches via registry
- [x] `build_gradient_mask` preserved as wrapper (backward compat)
- [x] `flatten_observed_gradients` in public API (used by rank_reconstruction_plot + jacobian)
- [x] `_get_last_fc_param_indices` in public API (used by run_single_exp.py)
- [x] `functions/masking.py` is a shim (all existing callers unmodified)
- [x] Golden tests guard all 15 mask modes — hashes must not change
- [x] `stable_ginv/metrics/jacobian.py` updated to use new import path
- [x] Categorized commits: `refactor:` / `test:` / `docs:`
- [x] HANDOVER_RESTRUCTURE.md updated

**Potential issues to watch:**
- `prefix_layer_fracs` defaults to `{}` in `Masker.__init__`. The `_PrefixTopKStrategy` calls `{k: int(v) for k, v in prefix_layer_fracs.items()}` — this is safe with an empty dict.
- `gradsize_topk_entries_layer` uses ALL param names (no VGG exclusion), while `gradsize_topfrac_entries_layer` uses `_gradsize_entries_layer_param_names` (excludes VGG classifier). This asymmetry is intentional and matches the original code exactly — do not "fix" it.
- The `else` fallback in `Masker.apply` (for modes not in `STRATEGY_REGISTRY`) calls `get_keep_ids(mask_mode, net=net)`. This handles `"all"` mode and raises `ValueError` for any truly unknown mode — same behavior as the original.
