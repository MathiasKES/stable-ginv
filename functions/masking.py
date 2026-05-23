import torch
import torch.nn as nn


def _get_last_fc_param_indices(net):
    """Return the parameter indices (weight + bias) of the last nn.Linear in the network."""
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


def _grad_magnitude(g, metric):
    """Scalar gradient magnitude for a single tensor using the given metric."""
    if metric == "l2":       return g.detach().norm(p=2).item()
    if metric == "mean_abs": return g.detach().abs().mean().item()
    if metric == "sum_abs":  return g.detach().abs().sum().item()
    raise ValueError(f"Unknown metric: {metric}")


def flatten_observed_gradients(grad_list, keep_ids=None, entry_masks=None):
    """Flatten selected gradient entries; keep_ids and entry_masks are mutually exclusive."""
    flat = []
    for i, g in enumerate(grad_list):
        if g is None:
            continue
        if entry_masks is not None:
            m = entry_masks[i]
            if m is None:
                continue
            flat.append(g.reshape(-1)[m.reshape(-1)])
        else:
            if keep_ids is not None and i not in keep_ids:
                continue
            flat.append(g.reshape(-1))
    if len(flat) == 0:
        raise ValueError("No observed gradients selected.")
    return torch.cat(flat, dim=0)


def get_keep_ids_by_gradsize(original_dy_dx, mode="topk", topk=10, top_frac=None, threshold=None, metric="l2", candidate_ids=None):
    """Tensor-wise masking: return (sorted keep_ids, sizes_sorted) by gradient magnitude."""
    sizes = []
    for i, g in enumerate(original_dy_dx):
        if candidate_ids is not None and i not in candidate_ids:
            continue
        if g is None:
            sizes.append((i, float("-inf")))
            continue

        sizes.append((i, _grad_magnitude(g, metric)))

    sizes_sorted = sorted(sizes, key=lambda x: x[1], reverse=True)

    if mode == "topk":
        k = min(topk, len(sizes_sorted))
        keep = [i for i, _ in sizes_sorted[:k]]

    elif mode == "topfrac":
        if top_frac is None:
            raise ValueError("top_frac must be set for mode='topfrac'")
        k = max(1, int(round(top_frac * len(sizes_sorted))))
        keep = [i for i, _ in sizes_sorted[:k]]

    elif mode == "threshold":
        if threshold is None:
            raise ValueError("threshold must be set for mode='threshold'")
        keep = [i for i, s in sizes_sorted if s >= threshold]
        if len(keep) == 0:
            keep = [sizes_sorted[0][0]]

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return sorted(keep), sizes_sorted


def get_entry_masks_by_gradsize(original_dy_dx, mode="topk_entries", topk=None, top_frac=None, candidate_ids=None):
    """Entry-wise masking by global gradient magnitude; returns (entry_masks, kept, total)."""
    pieces = []
    meta = []

    for i, g in enumerate(original_dy_dx):
        if candidate_ids is not None and i not in candidate_ids:
            meta.append((i, 0, None))
            continue
        if g is None:
            meta.append((i, 0, None))
            continue
        flat = g.detach().abs().reshape(-1)
        pieces.append(flat)
        meta.append((i, flat.numel(), g.shape))

    if len(pieces) == 0:
        raise ValueError("No gradients available for masking.")

    all_vals = torch.cat(pieces, dim=0)
    total_entries = all_vals.numel()

    if mode == "topk_entries":
        if topk is None:
            raise ValueError("topk must be set for mode='topk_entries'")
        k = max(1, min(int(topk), total_entries))
    elif mode == "topfrac_entries":
        if top_frac is None:
            raise ValueError("top_frac must be set for mode='topfrac_entries'")
        k = max(1, min(int(round(top_frac * total_entries)), total_entries))
    else:
        raise ValueError(f"Unknown entrywise mode: {mode}")

    top_idx = torch.topk(all_vals, k=k, largest=True).indices
    global_mask = torch.zeros(total_entries, dtype=torch.bool, device=all_vals.device)
    global_mask[top_idx] = True

    entry_masks = []
    offset = 0
    for i, numel, shape in meta:
        g = original_dy_dx[i]

        if g is None or shape is None or numel == 0:
            entry_masks.append(None)
            continue

        local_mask = global_mask[offset:offset + numel].reshape(shape)
        entry_masks.append(local_mask)
        offset += numel

    return entry_masks, k, total_entries


def get_entry_masks_by_prefix_group(
    net,
    original_dy_dx,
    prefixes,
    mode="topfrac_entries",
    topk=None,
    top_frac=None,
    prefix_top_fracs=None
):
    """Entry-wise masking ranked within each prefix group; returns (entry_masks, kept, total)."""
    named_params = list(net.named_parameters())
    entry_masks = [None] * len(original_dy_dx)

    kept_entries = 0
    total_entries = 0

    if prefix_top_fracs is None:
        prefix_top_fracs = {}

    for prefix in prefixes:
        group_infos = []
        pieces = []

        for i, (name, _) in enumerate(named_params):
            if not (name == prefix or name.startswith(prefix + ".")):
                continue

            g = original_dy_dx[i]
            if g is None:
                continue

            flat = g.detach().abs().reshape(-1)
            group_infos.append((i, g.shape, flat.numel()))
            pieces.append(flat)

        if len(pieces) == 0:
            continue

        all_vals = torch.cat(pieces, dim=0)
        n = all_vals.numel()
        total_entries += n

        if mode == "topk_entries":
            if topk is None:
                raise ValueError("topk must be set for mode='topk_entries'")
            k = max(1, min(int(topk), n))
        elif mode == "topfrac_entries":
            local_top_frac = prefix_top_fracs.get(prefix, top_frac)
            if local_top_frac is None:
                raise ValueError("top_frac must be set for mode='topfrac_entries'")
            k = max(1, min(int(round(local_top_frac * n)), n))
        else:
            raise ValueError(f"Unknown mode: {mode}")

        top_idx = torch.topk(all_vals, k=k, largest=True).indices
        group_mask = torch.zeros(n, dtype=torch.bool, device=all_vals.device)
        group_mask[top_idx] = True

        offset = 0
        for i, shape, numel in group_infos:
            local_mask = group_mask[offset:offset + numel].reshape(shape)
            entry_masks[i] = local_mask
            offset += numel

        kept_entries += k

    if kept_entries == 0:
        raise ValueError(f"No gradients matched prefixes={prefixes}")

    return entry_masks, kept_entries, total_entries


def get_keep_ids_by_prefix_group(
    net,
    original_dy_dx,
    prefixes,
    mode="topfrac",
    topk=None,
    top_frac=None,
    prefix_top_fracs=None,
    prefix_top_ks=None,
    metric="l2",
):
    """Tensor-wise masking ranked within each prefix group; returns (sorted keep_ids, ranked_by_prefix)."""
    if prefix_top_fracs is None:
        prefix_top_fracs = {}

    if prefix_top_ks is None:
        prefix_top_ks = {}

    named_params = list(net.named_parameters())
    keep = set()
    ranked_by_prefix = {}

    for prefix in prefixes:
        sizes = []

        for i, (name, _) in enumerate(named_params):
            if not (name == prefix or name.startswith(prefix + ".")):
                continue

            g = original_dy_dx[i]
            if g is None:
                continue

            sizes.append((i, _grad_magnitude(g, metric)))

        if len(sizes) == 0:
            continue

        sizes_sorted = sorted(sizes, key=lambda x: x[1], reverse=True)
        ranked_by_prefix[prefix] = sizes_sorted

        if mode == "topk":
            local_topk = prefix_top_ks.get(prefix, topk)
            if local_topk is None:
                raise ValueError("topk must be set for mode='topk'")
            k = max(0, min(int(local_topk), len(sizes_sorted)))

        elif mode == "topfrac":
            local_top_frac = prefix_top_fracs.get(prefix, top_frac)
            if local_top_frac is None:
                raise ValueError("top_frac must be set for mode='topfrac'")
            k = max(1, min(int(round(local_top_frac * len(sizes_sorted))), len(sizes_sorted)))

        else:
            raise ValueError(f"Unknown mode: {mode}")

        if k > 0:
            keep.update(i for i, _ in sizes_sorted[:k])

    if len(keep) == 0:
        raise ValueError(f"No parameters matched prefixes={prefixes}")

    return sorted(keep), ranked_by_prefix


def get_prefix_keep_ids(net, prefixes):
    """Return set of parameter indices whose name starts with any of the given prefixes."""
    keep = set()
    for idx, (name, _) in enumerate(net.named_parameters()):
        if any(name == p or name.startswith(p + ".") for p in prefixes):
            keep.add(idx)
    if len(keep) == 0:
        raise ValueError(f"No parameters matched prefixes={prefixes}")
    return keep


def get_keep_ids(mask_mode: str, net=None, prefixes=None):
    """Return set of parameter indices to keep for the given mask_mode."""
    if prefixes is not None:
        if net is None:
            raise ValueError("prefix-based keep_ids requires 'net'")
        return get_prefix_keep_ids(net, prefixes)

    if mask_mode == "all":
        if net is None:
            return set(range(8))
        return set(range(len(list(net.parameters()))))

    raise ValueError(f"Unknown MASK_MODE: {mask_mode}")


def build_gradient_mask(
    method,
    mask_mode,
    net,
    original_dy_dx,
    prefixes=(),
    prefix_layer_fracs=None,
    gradsize_topk=20,
    gradsize_topfrac=0.5,
    gradsize_threshold=None,
    gradsize_metric="l2",
):
    """Dispatch to the appropriate masking strategy; returns (keep_ids, entry_masks), exactly one None."""
    candidate_ids = None
    keep_ids = None
    entry_masks = None

    if prefix_layer_fracs is None:
        prefix_layer_fracs = {}

    if method == "idlg":
        keep_ids = get_keep_ids("all", net=net)
        return keep_ids, entry_masks

    if mask_mode.startswith("prefix_"):
        candidate_ids = get_prefix_keep_ids(net, prefixes)

    if mask_mode == "gradsize_topk":
        keep_ids, _ = get_keep_ids_by_gradsize(
            original_dy_dx, mode="topk", topk=gradsize_topk, metric=gradsize_metric
        )
    elif mask_mode == "gradsize_topfrac":
        keep_ids, _ = get_keep_ids_by_gradsize(
            original_dy_dx, mode="topfrac", top_frac=gradsize_topfrac, metric=gradsize_metric
        )
    elif mask_mode == "gradsize_topk_entries":
        entry_masks, _, _ = get_entry_masks_by_gradsize(
            original_dy_dx, mode="topk_entries", topk=gradsize_topk
        )
    elif mask_mode == "gradsize_topfrac_entries":
        entry_masks, _, _ = get_entry_masks_by_gradsize(
            original_dy_dx, mode="topfrac_entries", top_frac=gradsize_topfrac
        )
    elif mask_mode == "prefix_topk":
        keep_ids, _ = get_keep_ids_by_prefix_group(
            net=net,
            original_dy_dx=original_dy_dx,
            prefixes=prefixes,
            mode="topk",
            topk=gradsize_topk,
            prefix_top_ks={k: int(v) for k, v in prefix_layer_fracs.items()},
            metric=gradsize_metric,
        )
    elif mask_mode == "prefix_topfrac":
        keep_ids, _ = get_keep_ids_by_gradsize(
            original_dy_dx, mode="topfrac", top_frac=gradsize_topfrac,
            metric=gradsize_metric, candidate_ids=candidate_ids
        )
    elif mask_mode == "prefix_topk_entries":
        entry_masks, _, _ = get_entry_masks_by_gradsize(
            original_dy_dx, mode="topk_entries", topk=gradsize_topk,
            candidate_ids=candidate_ids
        )
    elif mask_mode == "prefix_topfrac_entries":
        entry_masks, _, _ = get_entry_masks_by_gradsize(
            original_dy_dx, mode="topfrac_entries", top_frac=gradsize_topfrac,
            candidate_ids=candidate_ids
        )
    elif mask_mode == "prefix_topk_entries_layer":
        entry_masks, _, _ = get_entry_masks_by_prefix_group(
            net=net,
            original_dy_dx=original_dy_dx,
            prefixes=prefixes,
            mode="topk_entries",
            topk=gradsize_topk,
            prefix_top_fracs=prefix_layer_fracs,
        )
    elif mask_mode == "prefix_topfrac_entries_layer":
        entry_masks, _, _ = get_entry_masks_by_prefix_group(
            net=net,
            original_dy_dx=original_dy_dx,
            prefixes=prefixes,
            mode="topfrac_entries",
            top_frac=gradsize_topfrac,
            prefix_top_fracs=prefix_layer_fracs,
        )
    elif mask_mode == "gradsize_topfrac_entries_layer":
        # Each parameter tensor is its own group → per-tensor independent selection.
        # Equivalent to prefix_topfrac_entries_layer with every param name as its own prefix.
        all_param_names = tuple(name for name, _ in net.named_parameters())
        entry_masks, _, _ = get_entry_masks_by_prefix_group(
            net=net, original_dy_dx=original_dy_dx, prefixes=all_param_names,
            mode="topfrac_entries", top_frac=gradsize_topfrac,
        )
    elif mask_mode == "gradsize_topk_entries_layer":
        all_param_names = tuple(name for name, _ in net.named_parameters())
        entry_masks, _, _ = get_entry_masks_by_prefix_group(
            net=net, original_dy_dx=original_dy_dx, prefixes=all_param_names,
            mode="topk_entries", topk=gradsize_topk,
        )
    elif mask_mode == "gradsize_threshold":
        keep_ids, _ = get_keep_ids_by_gradsize(
            original_dy_dx, mode="threshold", threshold=gradsize_threshold,
            metric=gradsize_metric
        )
    elif mask_mode == "prefix":
        keep_ids = get_keep_ids(mask_mode="prefix", net=net, prefixes=prefixes)
    else:
        keep_ids = get_keep_ids(mask_mode, net=net)

    # Always preserve the last FC layer so iDLG label inference is never blocked.
    last_fc_ids = _get_last_fc_param_indices(net)
    if keep_ids is not None:
        keep_ids = set(keep_ids) | last_fc_ids
    elif entry_masks is not None:
        for i in last_fc_ids:
            if i < len(entry_masks) and i < len(original_dy_dx) and original_dy_dx[i] is not None:
                entry_masks[i] = torch.ones(original_dy_dx[i].shape, dtype=torch.bool,
                                            device=original_dy_dx[i].device)

    return keep_ids, entry_masks
