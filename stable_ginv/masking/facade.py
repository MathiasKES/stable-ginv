"""Masker facade: resolves a (method, mask_mode) pair to a strategy and applies it."""
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
        """Store the selection inputs that govern which strategy is chosen.

        Parameters
        ----------
        method : str
            Reconstruction method name (e.g. ``"idlg"``).  When ``"idlg"``,
            no masking is applied regardless of ``mask_mode``.
        mask_mode : str
            Key into ``STRATEGY_REGISTRY`` that selects the masking strategy
            (e.g. ``"gradsize_topfrac"``).  The value ``"all"`` keeps every
            parameter; other unknown values are forwarded to ``get_keep_ids``.
        prefixes : tuple, optional
            Parameter-name prefixes that restrict the candidate set for
            prefix-family strategies.  Defaults to ``()`` (no restriction).
        prefix_layer_fracs : dict or None, optional
            Per-prefix keep fractions or counts used by prefix-layer strategies.
            ``None`` is normalised to an empty dict.
        gradsize_topk : int, optional
            Number of top parameters or entries to keep for top-k strategies.
            Default is ``20``.
        gradsize_topfrac : float, optional
            Fraction of parameters or entries to keep for top-fraction
            strategies.  Default is ``0.5``.
        gradsize_metric : str, optional
            Gradient-magnitude metric used for ranking (e.g. ``"l2"``).
            Default is ``"l2"``.
        """
        self._method = method
        self._mask_mode = mask_mode
        self._prefixes = prefixes
        self._prefix_layer_fracs = prefix_layer_fracs or {}
        self._gradsize_topk = gradsize_topk
        self._gradsize_topfrac = gradsize_topfrac
        self._gradsize_metric = gradsize_metric

    def apply(self, net, original_dy_dx):
        """Select the appropriate strategy and run it against the network and observed gradients.

        Parameters
        ----------
        net : torch.nn.Module
            The model whose parameter list defines the gradient index space.
        original_dy_dx : sequence of torch.Tensor
            Observed gradients, one tensor per named parameter of ``net``.

        Returns
        -------
        keep_ids : list or None
            Indices of whole parameters to retain.  Non-None for
            whole-parameter strategies; ``None`` for entry-level strategies.
        entry_masks : list of torch.Tensor or None
            Boolean masks, one per parameter, marking individual gradient
            entries to retain.  Non-None for entry-level strategies; ``None``
            for whole-parameter strategies.
        """
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
