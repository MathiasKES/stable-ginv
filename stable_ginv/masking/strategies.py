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
    "gradsize_topk":                  _GradsizeTopKStrategy,
    "gradsize_topfrac":               _GradsizeTopFracStrategy,
    "gradsize_topk_entries":          _GradsizeTopKEntriesStrategy,
    "gradsize_topfrac_entries":       _GradsizeTopFracEntriesStrategy,
    "gradsize_topk_entries_layer":    _GradsizeTopKEntriesLayerStrategy,
    "gradsize_topfrac_entries_layer": _GradsizeTopFracEntriesLayerStrategy,
    "prefix_topk":                    _PrefixTopKStrategy,
    "prefix_topfrac":                 _PrefixTopFracStrategy,
    "prefix_topk_entries":            _PrefixTopKEntriesStrategy,
    "prefix_topfrac_entries":         _PrefixTopFracEntriesStrategy,
    "prefix_topk_entries_layer":      _PrefixTopKEntriesLayerStrategy,
    "prefix_topfrac_entries_layer":   _PrefixTopFracEntriesLayerStrategy,
    "prefix":                         _PrefixStrategy,
}
