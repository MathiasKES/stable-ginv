import torch
import torch.nn as nn
import pytest

from functions.masking import (
    flatten_observed_gradients,
    get_keep_ids_by_gradsize,
    get_entry_masks_by_gradsize,
    build_gradient_mask,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def make_grads(sizes):
    """Tensors filled with float(size) so L2 norm order matches size order."""
    return [torch.full((s,), float(s)) for s in sizes]


def small_net():
    """2-layer linear net with 4 named parameters: 0.weight, 0.bias, 1.weight, 1.bias."""
    return nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))


def small_grads(net):
    """Gradient list matching small_net's parameter shapes, filled with 1s."""
    return [torch.ones_like(p) for p in net.parameters()]


# ── flatten_observed_gradients ────────────────────────────────────────────────

def test_flatten_all_gradients():
    grads = [torch.ones(3), torch.ones(4), torch.ones(5)]
    out = flatten_observed_gradients(grads)
    assert out.shape == (12,)


def test_flatten_with_keep_ids():
    grads = [torch.ones(3), torch.ones(4), torch.ones(5)]
    out = flatten_observed_gradients(grads, keep_ids={0, 2})
    assert out.shape == (8,)  # 3 + 5; tensor 1 excluded


def test_flatten_with_entry_masks():
    grads = [torch.ones(6), torch.ones(4)]
    # mask for tensor 0: keep first 3; tensor 1: None → skipped
    masks = [torch.tensor([True, True, True, False, False, False]), None]
    out = flatten_observed_gradients(grads, entry_masks=masks)
    assert out.shape == (3,)


def test_flatten_skips_none_gradients():
    grads = [torch.ones(3), None, torch.ones(5)]
    out = flatten_observed_gradients(grads)
    assert out.shape == (8,)  # None gradient at index 1 skipped


def test_flatten_raises_when_nothing_selected():
    grads = [torch.ones(4)]
    with pytest.raises(ValueError):
        flatten_observed_gradients(grads, keep_ids=set())  # empty keep_ids


# ── get_keep_ids_by_gradsize ──────────────────────────────────────────────────

def test_topk_returns_highest_norm_indices():
    # L2 norms: tensor at idx 3 (value=10) > idx 1 (value=5) > idx 2 (value=2) > idx 0 (value=1)
    grads = [torch.full((1,), 1.0), torch.full((1,), 5.0),
             torch.full((1,), 2.0), torch.full((1,), 10.0)]
    ids, _ = get_keep_ids_by_gradsize(grads, mode="topk", topk=2)
    assert set(ids) == {1, 3}


def test_topfrac_returns_correct_count():
    grads = make_grads([1, 2, 3, 4])  # 4 tensors
    ids, _ = get_keep_ids_by_gradsize(grads, mode="topfrac", top_frac=0.5)
    assert len(ids) == 2


def test_threshold_keeps_above_value():
    grads = [torch.full((1,), 1.0), torch.full((1,), 10.0), torch.full((1,), 5.0)]
    ids, _ = get_keep_ids_by_gradsize(grads, mode="threshold", threshold=4.0)
    assert set(ids) == {1, 2}  # norms 10 and 5 are >= 4


def test_threshold_fallback_when_none_qualify():
    # threshold so high nothing passes → should fall back to keeping the 1 largest
    grads = [torch.full((1,), 1.0), torch.full((1,), 2.0)]
    ids, _ = get_keep_ids_by_gradsize(grads, mode="threshold", threshold=100.0)
    assert len(ids) == 1
    assert set(ids) == {1}  # largest is idx 1


def test_none_gradients_ranked_last():
    # None gradients get -inf magnitude so they sort after all real gradients
    grads = [None, torch.full((1,), 5.0), None]
    ids, _ = get_keep_ids_by_gradsize(grads, mode="topk", topk=1)
    assert set(ids) == {1}  # topk=1 → only the non-None gradient selected


# ── get_entry_masks_by_gradsize ───────────────────────────────────────────────

def test_entry_topk_keeps_exact_k_entries():
    grads = [torch.arange(1, 6, dtype=torch.float),   # 5 entries: [1,2,3,4,5]
             torch.arange(1, 5, dtype=torch.float)]   # 4 entries: [1,2,3,4]
    masks, kept, total = get_entry_masks_by_gradsize(grads, mode="topk_entries", topk=3)
    assert total == 9
    assert kept == 3
    true_count = sum(m.sum().item() for m in masks if m is not None)
    assert true_count == 3


def test_entry_topfrac_keeps_correct_fraction():
    grads = [torch.ones(10)]
    masks, kept, total = get_entry_masks_by_gradsize(grads, mode="topfrac_entries", top_frac=0.4)
    assert total == 10
    assert kept == 4
    assert masks[0].sum().item() == 4


def test_entry_masks_shape_matches_gradient():
    grads = [torch.ones(3, 4), torch.ones(5)]
    masks, _, _ = get_entry_masks_by_gradsize(grads, mode="topfrac_entries", top_frac=0.5)
    assert masks[0].shape == (3, 4)
    assert masks[1].shape == (5,)


def test_entry_candidate_ids_restricts_pool():
    grads = [torch.ones(5), torch.ones(5), torch.ones(5)]
    masks, _, _ = get_entry_masks_by_gradsize(
        grads, mode="topfrac_entries", top_frac=1.0, candidate_ids={1}
    )
    assert masks[0] is None   # not in candidate_ids
    assert masks[2] is None   # not in candidate_ids
    assert masks[1] is not None
    assert masks[1].all()     # top_frac=1.0 → all entries in candidate kept


# ── build_gradient_mask routing ───────────────────────────────────────────────

def test_idlg_method_keeps_all_params():
    net = small_net()
    grads = small_grads(net)
    keep_ids, entry_masks = build_gradient_mask("idlg", "gradsize_topk", net, grads)
    assert entry_masks is None
    assert keep_ids == set(range(4))  # all 4 parameters kept


def test_gradsize_topk_returns_keep_ids_not_entry_masks():
    net = small_net()
    grads = [torch.full((1,), float(v)) for v in [1, 5, 2, 10]]
    keep_ids, entry_masks = build_gradient_mask(
        "masked", "gradsize_topk", net, grads, gradsize_topk=2
    )
    assert entry_masks is None
    assert len(keep_ids) == 2


def test_gradsize_topfrac_returns_keep_ids():
    net = small_net()
    grads = make_grads([1, 2, 3, 4])
    keep_ids, entry_masks = build_gradient_mask(
        "masked", "gradsize_topfrac", net, grads, gradsize_topfrac=0.5
    )
    assert entry_masks is None
    assert len(keep_ids) == 2


def test_gradsize_topfrac_entries_returns_entry_masks_not_keep_ids():
    net = small_net()
    grads = [torch.ones(12), torch.ones(3), torch.ones(6), torch.ones(2)]
    keep_ids, entry_masks = build_gradient_mask(
        "masked", "gradsize_topfrac_entries", net, grads, gradsize_topfrac=0.5
    )
    assert keep_ids is None
    assert entry_masks is not None
    assert len(entry_masks) == 4


def test_prefix_mode_keeps_only_matching_layer():
    net = small_net()
    grads = small_grads(net)
    # Named params: "0.weight" (idx 0), "0.bias" (idx 1), "1.weight" (idx 2), "1.bias" (idx 3)
    keep_ids, entry_masks = build_gradient_mask(
        "masked", "prefix", net, grads, prefixes=("0",)
    )
    assert entry_masks is None
    assert keep_ids == {0, 1}  # only layer "0" params
