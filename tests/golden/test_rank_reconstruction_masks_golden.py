"""Golden test: the rank-vs-reconstruction entry-mask builder is byte-stable.

`_build_entry_masks` (topk_abs path) selects the top-|grad| entries that define
each reconstruction budget — globally (``exclude_fc=False``) or over non-FC
params only (``exclude_fc=True``, the --keep_fc path). Synthetic CPU grads make
the selected indices deterministic. ``net`` is unused on the topk_abs path, so
``None`` is passed here.
"""
import torch

from stable_ginv.viz.plot_rank_reconstruction import _build_entry_masks
from tests.golden.helpers import load_or_regen


def _fixed_grads():
    # Four "parameter" tensors with distinct, deterministic magnitudes.
    return [
        torch.tensor([0.5, -2.0, 1.0, 0.1]),
        torch.tensor([[3.0, -0.2], [0.05, 4.0]]),
        torch.tensor([-1.5, 0.3, 2.5]),
        torch.tensor([0.9, -0.9]),  # treated as the FC tensor below
    ]


def _masks_to_indices(entry_masks):
    """Serialize a list of bool masks to {param_index: sorted true flat-indices}."""
    out = {}
    for i, m in enumerate(entry_masks):
        if m is None:
            continue
        idx = torch.nonzero(m.reshape(-1), as_tuple=False).reshape(-1).tolist()
        out[str(i)] = sorted(int(j) for j in idx)
    return out


def _produce():
    grads = _fixed_grads()
    fc_ids = {3}  # last tensor is the FC layer
    return {
        "global_topk_b6": _masks_to_indices(
            _build_entry_masks(grads, set(), 6, "topk_abs", None, False)
        ),
        "global_topk_b3": _masks_to_indices(
            _build_entry_masks(grads, set(), 3, "topk_abs", None, False)
        ),
        "keepfc_topk_b4": _masks_to_indices(
            _build_entry_masks(grads, fc_ids, 4, "topk_abs", None, True)
        ),
    }


def test_rank_reconstruction_masks_stable():
    golden = load_or_regen("rank_reconstruction_masks_golden.json", _produce)
    assert _produce() == golden
