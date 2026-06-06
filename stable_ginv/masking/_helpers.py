"""Internal masking helpers: parameter-index lookups and gradient-magnitude scoring."""
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
