"""Golden test: masked registry keys are stable across all mask modes.

Guards stable_ginv.registry.masked_key_from_args against accidental changes to the
comparable-args dict or its JSON serialization during the Phase 4 registry split.
"""
from types import SimpleNamespace

from stable_ginv.registry import masked_key_from_args
from tests.golden.helpers import load_or_regen

MASK_MODES = [
    "gradsize_topk", "gradsize_topfrac", "gradsize_topk_entries",
    "gradsize_topfrac_entries", "gradsize_topk_entries_layer",
    "gradsize_topfrac_entries_layer", "prefix", "prefix_topk", "prefix_topfrac",
    "prefix_topk_entries", "prefix_topfrac_entries", "prefix_topk_entries_layer",
    "prefix_topfrac_entries_layer", "none",
]


def _args(mask_mode):
    return SimpleNamespace(
        dataset="cifar100", network="vgg13", pretrained=False, lr=0.1, gamma=0.5,
        grad_loss="cos", num_dummy=1, iteration=5000, tv_weight=0.0,
        optimizer="signed_adamw", num_restarts=1, max_iteration=20, history_size=100,
        mask_mode=mask_mode, gradsize_topk=None, gradsize_topfrac=0.1,
        gradsize_metric="abs", prefixes="conv1:0.5,fc", num_exp=30, run_id=0,
    )


def _produce():
    return {mode: masked_key_from_args(_args(mode))[0] for mode in MASK_MODES}


def test_masked_keys_stable_across_modes():
    golden = load_or_regen("registry_key_golden.json", _produce)
    current = _produce()
    for mode, expected in golden.items():
        assert current[mode] == expected, (
            f"Registry key changed for mask_mode={mode!r}: "
            f"golden={expected!r} current={current[mode]!r}"
        )
