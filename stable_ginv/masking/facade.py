from stable_ginv.masking.strategies import _IdlgStrategy, STRATEGY_REGISTRY
from stable_ginv.masking._compute import get_keep_ids


class Masker:
    """Selects and applies the right masking strategy from STRATEGY_REGISTRY."""

    def __init__(
        self,
        method: str,
        mask_mode: str,
        prefixes: tuple = (),
        prefix_layer_fracs: dict | None = None,
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
