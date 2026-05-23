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
    assert len(keep_ids) >= 2
    # last FC layer (layer "1", indices 2 and 3) must always be present
    assert {2, 3}.issubset(keep_ids)


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
    # prefix "0" selects {0, 1}; last FC (layer "1", indices 2, 3) always force-included
    assert keep_ids == {0, 1, 2, 3}


def test_last_fc_always_preserved_in_topk():
    """Last FC layer must appear in keep_ids even when topk would exclude it."""
    net = small_net()
    # Give layer "1" (last FC, indices 2,3) the lowest norms so topk=1 would normally exclude it
    grads = [torch.full((1,), 10.0), torch.full((1,), 9.0),   # layer 0: high norms
             torch.full((1,), 0.1),  torch.full((1,), 0.1)]   # layer 1 (last FC): low norms
    keep_ids, _ = build_gradient_mask("masked", "gradsize_topk", net, grads, gradsize_topk=1)
    assert {2, 3}.issubset(keep_ids)  # last FC forced in despite low norm


def test_last_fc_always_preserved_in_entry_masks():
    """Last FC layer must be fully unmasked (all-True) even in entry-wise masking."""
    net = small_net()
    grads = [torch.ones(12), torch.ones(3), torch.ones(6), torch.ones(2)]
    _, entry_masks = build_gradient_mask(
        "masked", "gradsize_topfrac_entries", net, grads, gradsize_topfrac=0.1
    )
    assert entry_masks is not None
    # indices 2 and 3 are last FC — their masks must be all True
    assert entry_masks[2].all()
    assert entry_masks[3].all()


# ── gradsize_topfrac_entries_layer / gradsize_topk_entries_layer ──────────────
# These modes delegate to get_entry_masks_by_prefix_group with every param name
# as its own prefix, giving per-tensor independent selection.

def test_per_layer_topfrac_keeps_correct_fraction_each_layer():
    # small_net: 0.weight=12, 0.bias=3, 1.weight=6, 1.bias=2
    net = small_net()
    grads = [torch.arange(1, 13, dtype=torch.float),   # 0.weight: 12 entries
             torch.arange(1, 4,  dtype=torch.float),   # 0.bias:    3 entries
             torch.arange(1, 7,  dtype=torch.float),   # 1.weight:  6 entries (last FC)
             torch.arange(1, 3,  dtype=torch.float)]   # 1.bias:    2 entries (last FC)
    _, masks = build_gradient_mask("masked", "gradsize_topfrac_entries_layer", net, grads, gradsize_topfrac=0.5)
    assert masks[0].sum().item() == 6   # 50% of 12
    assert masks[1].sum().item() == 2   # 50% of 3 (rounds to 2)
    assert masks[2].all()               # last FC — always fully unmasked
    assert masks[3].all()               # last FC — always fully unmasked


def test_per_layer_each_layer_independent():
    # Non-FC layer has tiny values; global topfrac would deprioritise it.
    # Per-layer mode must still keep 50% of its own entries.
    net = small_net()
    grads = [torch.full((12,), 0.001),  # 0.weight — tiny values
             torch.full((3,),  0.001),  # 0.bias   — tiny values
             torch.full((6,),  100.0),  # 1.weight — large (last FC)
             torch.full((2,),  100.0)]  # 1.bias   — large (last FC)
    _, masks = build_gradient_mask("masked", "gradsize_topfrac_entries_layer", net, grads, gradsize_topfrac=0.5)
    assert masks[0].sum().item() == 6   # 50% of tiny layer still kept
    assert masks[1].sum().item() == 2   # 50% of tiny layer still kept
    assert masks[2].all()
    assert masks[3].all()


def test_per_layer_shape_preserved():
    net = small_net()
    grads = small_grads(net)
    _, masks = build_gradient_mask("masked", "gradsize_topfrac_entries_layer", net, grads, gradsize_topfrac=0.5)
    for p, m in zip(net.parameters(), masks):
        assert m.shape == p.shape


def test_per_layer_none_gradient_skipped():
    # Gradient at index 1 (0.bias) is None — its mask must remain None.
    net = small_net()
    grads = [torch.ones(12), None, torch.ones(6), torch.ones(2)]
    _, masks = build_gradient_mask("masked", "gradsize_topfrac_entries_layer", net, grads, gradsize_topfrac=0.5)
    assert masks[1] is None
    assert masks[0] is not None


def test_per_layer_mode_via_build_gradient_mask():
    net = small_net()
    grads = [torch.arange(1, 13, dtype=torch.float),  # 12 entries (0.weight)
             torch.ones(3),                            # 3 entries  (0.bias)
             torch.arange(1, 7,  dtype=torch.float),  # 6 entries  (1.weight)
             torch.ones(2)]                            # 2 entries  (1.bias)
    keep_ids, entry_masks = build_gradient_mask(
        "masked", "gradsize_topfrac_entries_layer", net, grads, gradsize_topfrac=0.5
    )
    assert keep_ids is None
    assert entry_masks is not None
    # Each layer keeps 50% of its own entries
    assert entry_masks[0].sum().item() == 6   # 50% of 12
    assert entry_masks[1].sum().item() == 2   # 50% of 3 (rounded up from 1.5)
    assert entry_masks[2].all()               # last FC — always fully unmasked
    assert entry_masks[3].all()               # last FC — always fully unmasked
