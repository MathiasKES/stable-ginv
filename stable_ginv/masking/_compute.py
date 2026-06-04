import torch

from stable_ginv.masking._helpers import _grad_magnitude


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


def get_keep_ids_by_gradsize(
    original_dy_dx, mode="topk", topk=10, top_frac=None, metric="l2", candidate_ids=None
):
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
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return sorted(keep), sizes_sorted


def get_entry_masks_by_gradsize(
    original_dy_dx, mode="topk_entries", topk=None, top_frac=None, candidate_ids=None
):
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
    prefix_top_fracs=None,
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
            if local_top_frac == 0:
                continue
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
            if local_top_frac == 0:
                k = 0
            else:
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


def get_keep_ids(mask_mode, net=None, prefixes=None):
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
