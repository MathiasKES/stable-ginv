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
