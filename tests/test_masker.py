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
    assert ids == set(range(len(list(net.parameters()))))
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
