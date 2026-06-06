"""Jacobian construction and numerical rank of the per-sample gradient map."""
import numpy as np
import torch
import torch.autograd.forward_ad as fwAD

from stable_ginv.masking import flatten_observed_gradients

_SCIPY_LINALG = None
_SCIPY_LINALG_IMPORT_ERROR = None


def _load_scipy_linalg():
    """Import scipy.linalg only when QR pivoting is requested; return None if unavailable.

    Mirrors the lazy scipy.stats loader in functions/io_utils.py. Normal
    reconstruction runs never need QR pivoting, so deferring this import keeps
    scipy.linalg out of worker startup, where an older system C++ runtime can
    make the import fail.
    """
    global _SCIPY_LINALG, _SCIPY_LINALG_IMPORT_ERROR
    if _SCIPY_LINALG is not None:
        return _SCIPY_LINALG
    if _SCIPY_LINALG_IMPORT_ERROR is not None:
        return None
    try:
        import scipy.linalg as scipy_linalg
    except Exception as exc:
        _SCIPY_LINALG_IMPORT_ERROR = exc
        return None
    _SCIPY_LINALG = scipy_linalg
    return _SCIPY_LINALG


def _get_layer_groups(net):
    """Map each parameter name to a group label for layer_spread.

    Top-level children become groups. A nn.Sequential is expanded one level
    when all of its direct children are leaf modules (no sub-children), e.g.
    VGG's features.0, features.2, ... Each ResNet BasicBlock/Bottleneck keeps
    its parent Sequential label (layer1, layer2, ...) because BasicBlocks have
    their own sub-modules.
    """
    groups = {}
    for child_name, child in net.named_children():
        if isinstance(child, torch.nn.Sequential):
            all_leaf = all(not list(gc.named_children()) for _, gc in child.named_children())
            if all_leaf:
                for gc_name, gc_module in child.named_children():
                    for param_name, _ in gc_module.named_parameters():
                        groups[f"{child_name}.{gc_name}.{param_name}"] = f"{child_name}.{gc_name}"
            else:
                for param_name, _ in child.named_parameters():
                    groups[f"{child_name}.{param_name}"] = child_name
        else:
            for param_name, _ in child.named_parameters():
                groups[f"{child_name}.{param_name}"] = child_name
    for param_name, _ in net.named_parameters(recurse=False):
        groups[param_name] = param_name
    return groups


def _layer_info(grads, keep_ids, entry_masks):
    """Return (sizes, ndims) for each tensor contributing to the flat gradient vector.

    ndim is the number of dimensions of the original parameter tensor:
      - ndim >= 2: weight tensors (conv, linear) — localized, informative for rank
      - ndim == 1: bias / BN scale / BN shift — small, diffuse, less informative
    """
    sizes, ndims = [], []
    for i, g in enumerate(grads):
        if g is None:
            continue
        if entry_masks is not None:
            m = entry_masks[i]
            if m is None:
                continue
            sizes.append(int(m.reshape(-1).sum().item()))
        else:
            if keep_ids is not None and i not in keep_ids:
                continue
            sizes.append(g.numel())
        ndims.append(g.ndim)
    return sizes, ndims


def _fc_flat_mask(grads, keep_ids, entry_masks, fc_param_indices, device):
    """Boolean mask over flatten_observed_gradients output; True where entry belongs to an FC param."""
    parts = []
    for i, g in enumerate(grads):
        if g is None:
            continue
        if entry_masks is not None:
            m = entry_masks[i]
            if m is None:
                continue
            n = int(m.reshape(-1).sum().item())
        else:
            if keep_ids is not None and i not in keep_ids:
                continue
            n = g.numel()
        parts.append(torch.full((n,), fill_value=(i in fc_param_indices), dtype=torch.bool, device=device))
    if not parts:
        return torch.zeros(0, dtype=torch.bool, device=device)
    return torch.cat(parts)


def _layer_spread_non_fc(g_obs_det, budget, grads, keep_ids, entry_masks, fc_param_indices, net):
    """layer_spread entry selection on non-FC params only; returns global flat indices into g_obs_det."""
    named_params = list(net.named_parameters())
    param_to_group = _get_layer_groups(net)

    group_totals = {}
    group_slices = {}
    group_order = []
    seen = set()
    global_offset = 0

    for i, (name, _) in enumerate(named_params):
        g = grads[i]
        if g is None:
            continue
        if entry_masks is not None:
            m = entry_masks[i]
            if m is None:
                continue
            n = int(m.reshape(-1).sum().item())
        else:
            if keep_ids is not None and i not in keep_ids:
                continue
            n = g.numel()

        if i not in fc_param_indices:
            grp = param_to_group.get(name, name.split('.')[0])
            if grp not in seen:
                group_order.append(grp)
                seen.add(grp)
                group_totals[grp] = 0
                group_slices[grp] = []
            group_totals[grp] += n
            group_slices[grp].append((global_offset, n))

        global_offset += n

    if not group_order:
        return torch.tensor([], dtype=torch.long, device=g_obs_det.device)

    k = budget
    num_groups = len(group_order)
    base = k // num_groups
    remainder = k % num_groups
    order = sorted(group_order, key=lambda grp: group_totals[grp], reverse=True)
    per_group = {grp: base for grp in group_order}
    for idx in range(remainder):
        per_group[order[idx]] += 1
    per_group = {grp: min(per_group[grp], group_totals[grp]) for grp in group_order}

    leftover = k - sum(per_group.values())
    if leftover > 0:
        for grp in order:
            if per_group[grp] < group_totals[grp]:
                give = min(leftover, group_totals[grp] - per_group[grp])
                per_group[grp] += give
                leftover -= give
                if leftover == 0:
                    break

    indices = []
    for grp in group_order:
        n = per_group[grp]
        slices = group_slices[grp]
        if n <= 0 or not slices:
            continue
        if n >= group_totals[grp]:
            for off, sz in slices:
                indices.append(torch.arange(sz, device=g_obs_det.device) + off)
        else:
            grp_vals = torch.cat([g_obs_det[off:off + sz] for off, sz in slices])
            top_local = torch.topk(grp_vals.abs(), k=n, largest=True).indices
            local_to_global = torch.cat([
                torch.arange(sz, device=g_obs_det.device) + off
                for off, sz in slices
            ])
            indices.append(local_to_global[top_local])

    return torch.cat(indices) if indices else torch.tensor([], dtype=torch.long, device=g_obs_det.device)


def _layer_dist_str(net, grads, keep_ids, entry_masks, selected_idx):
    """Return a compact string showing how many selected entries came from each parameter."""
    named = list(net.named_parameters())
    sel = selected_idx
    offset = 0
    counts = []
    for i, g in enumerate(grads):
        if g is None:
            continue
        if entry_masks is not None:
            m = entry_masks[i]
            if m is None:
                continue
            sz = int(m.reshape(-1).sum().item())
        else:
            if keep_ids is not None and i not in keep_ids:
                continue
            sz = g.numel()
        name = named[i][0]
        count = int(((sel >= offset) & (sel < offset + sz)).sum().item())
        counts.append((name, count, sz))
        offset += sz
    counts.sort(key=lambda x: -x[1])
    return "  ".join(f"{n}:{c}" for n, c, _ in counts if c > 0) or "(none)"


def _build_jacobian(net, x_norm, y, criterion, keep_ids, entry_masks,
                    max_entries, select_mode, device_for_J, progress_fn=None,
                    force_fc=False, fc_param_indices=None, print_layer_dist=False):
    """Forward pass + row selection + J construction.

    Automatically chooses the cheaper direction:
      - Forward-mode AD (column-wise, unknowns passes) when unknowns < used_entries.
      - Backward-mode AD (row-wise, used_entries passes) otherwise.

    Returns (J, total_entries, unknowns).
    J has shape (used_entries, unknowns).
    """
    params = tuple(net.parameters())
    x_norm = x_norm.detach()
    unknowns = x_norm.numel()

    # One backward pass (no create_graph) to get g_obs magnitudes for row selection.
    x_sel = x_norm.clone().requires_grad_(True)
    out = net(x_sel)
    loss = criterion(out, y)
    grads = torch.autograd.grad(loss, params, create_graph=False)
    g_obs_det = flatten_observed_gradients(grads, keep_ids=keep_ids, entry_masks=entry_masks).detach()
    total_entries = g_obs_det.numel()

    if force_fc and fc_param_indices:
        fc_bool = _fc_flat_mask(grads, keep_ids, entry_masks, fc_param_indices, g_obs_det.device)
        fc_idx = fc_bool.nonzero(as_tuple=True)[0]
        non_fc_idx = (~fc_bool).nonzero(as_tuple=True)[0]
        fc_size = fc_idx.numel()
        non_fc_total = non_fc_idx.numel()
        non_fc_k = non_fc_total if (max_entries is None or int(max_entries) >= non_fc_total) else int(max_entries)

        if non_fc_k == non_fc_total or non_fc_total == 0:
            non_fc_sel = non_fc_idx
        elif select_mode == "topk_abs":
            non_fc_sel = non_fc_idx[torch.topk(g_obs_det[non_fc_idx].abs(), k=non_fc_k, largest=True).indices]
        elif select_mode == "first":
            non_fc_sel = non_fc_idx[:non_fc_k]
        elif select_mode == "random":
            perm = torch.randperm(non_fc_total, device=g_obs_det.device)
            non_fc_sel = non_fc_idx[perm[:non_fc_k]]
        elif select_mode == "layer_spread":
            non_fc_sel = _layer_spread_non_fc(
                g_obs_det, non_fc_k, grads, keep_ids, entry_masks, fc_param_indices, net
            )
        else:
            raise ValueError(f"Unknown select_mode: {select_mode}")

        selected_idx = torch.cat([fc_idx, non_fc_sel])
        print(f"[keep_fc] non-FC={non_fc_k} + FC={fc_size} → total={selected_idx.numel()}")

    elif max_entries is None or max_entries >= total_entries:
        selected_idx = torch.arange(total_entries, device=g_obs_det.device)
    else:
        k = int(max_entries)
        if k <= 0:
            raise ValueError("max_entries must be positive")
        if select_mode == "topk_abs":
            selected_idx = torch.topk(g_obs_det.abs(), k=k, largest=True).indices
        elif select_mode == "first":
            selected_idx = torch.arange(k, device=g_obs_det.device)
        elif select_mode == "random":
            selected_idx = torch.randperm(total_entries, device=g_obs_det.device)[:k]
        elif select_mode == "layer_spread":
            layer_sizes, _ = _layer_info(grads, keep_ids, entry_masks)
            named_params = list(net.named_parameters())
            param_to_group = _get_layer_groups(net)

            group_of = []
            for i, (name, _p) in enumerate(named_params):
                if grads[i] is None:
                    continue
                if entry_masks is not None:
                    if entry_masks[i] is None:
                        continue
                else:
                    if keep_ids is not None and i not in keep_ids:
                        continue
                group_of.append(param_to_group.get(name, name.split('.')[0]))

            seen_grps = set()
            groups = []
            for grp in group_of:
                if grp not in seen_grps:
                    groups.append(grp)
                    seen_grps.add(grp)

            num_groups = len(groups)
            group_totals = {grp: 0 for grp in groups}
            group_slices = {grp: [] for grp in groups}
            offset = 0
            for sz, grp in zip(layer_sizes, group_of):
                group_totals[grp] += sz
                group_slices[grp].append((offset, sz))
                offset += sz

            base = k // num_groups
            remainder = k % num_groups
            order = sorted(groups, key=lambda grp: group_totals[grp], reverse=True)
            per_group = {grp: base for grp in groups}
            for i in range(remainder):
                per_group[order[i]] += 1
            per_group = {grp: min(per_group[grp], group_totals[grp]) for grp in groups}

            leftover = k - sum(per_group.values())
            if leftover > 0:
                for grp in order:
                    if per_group[grp] < group_totals[grp]:
                        give = min(leftover, group_totals[grp] - per_group[grp])
                        per_group[grp] += give
                        leftover -= give
                        if leftover == 0:
                            break

            indices = []
            for grp in groups:
                n = per_group[grp]
                slices = group_slices[grp]
                if n <= 0 or not slices:
                    continue
                if n >= group_totals[grp]:
                    for off, sz in slices:
                        indices.append(torch.arange(sz, device=g_obs_det.device) + off)
                else:
                    grp_vals = torch.cat([g_obs_det[off:off + sz] for off, sz in slices])
                    top_local = torch.topk(grp_vals.abs(), k=n, largest=True).indices
                    local_to_global = torch.cat([
                        torch.arange(sz, device=g_obs_det.device) + off
                        for off, sz in slices
                    ])
                    indices.append(local_to_global[top_local])

            selected_idx = (torch.cat(indices) if indices
                            else torch.arange(min(k, total_entries), device=g_obs_det.device))
        else:
            raise ValueError(f"Unknown select_mode: {select_mode}")

    used_entries = selected_idx.numel()
    if print_layer_dist:
        print(f"    [layer_dist entries={used_entries}] {_layer_dist_str(net, grads, keep_ids, entry_masks, selected_idx)}", flush=True)
    selected_idx_for_grad = selected_idx.to(x_norm.device)
    J = torch.empty((used_entries, unknowns), dtype=x_norm.dtype, device=device_for_J)

    if unknowns < used_entries:
        for j in range(unknowns):
            tangent = torch.zeros_like(x_norm)
            tangent.reshape(-1)[j] = 1.0
            with fwAD.dual_level():
                x_dual = fwAD.make_dual(x_norm, tangent)
                out_d = net(x_dual)
                loss_d = criterion(out_d, y)
                grads_d = torch.autograd.grad(loss_d, params)
                g_obs_d = flatten_observed_gradients(grads_d, keep_ids=keep_ids, entry_masks=entry_masks)
                g_col = g_obs_d[selected_idx_for_grad]
                col = fwAD.unpack_dual(g_col).tangent
                if col is None:
                    raise RuntimeError(
                        "Forward-mode AD did not propagate through autograd.grad. "
                        "Requires PyTorch >= 2.0. Upgrade PyTorch or open an issue."
                    )
                J[:, j] = col.detach().reshape(-1).to(device_for_J)
            if progress_fn is not None:
                progress_fn(1)
    else:
        x_req = x_norm.requires_grad_(True)
        out = net(x_req)
        loss = criterion(out, y)
        grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
        g_obs_full = flatten_observed_gradients(grads, keep_ids=keep_ids, entry_masks=entry_masks)
        g_sel = g_obs_full[selected_idx]

        for i in range(used_entries):
            grad_i = torch.autograd.grad(
                g_sel[i], x_req,
                retain_graph=(i < used_entries - 1),
                create_graph=False,
                allow_unused=False,
            )[0]
            J[i] = grad_i.reshape(-1).detach().to(device_for_J)
            if progress_fn is not None:
                progress_fn(1)

    return J, total_entries, unknowns


def _rank_of_J(J, print_svd_info=False, normalize_rows=True):
    """Compute numerical rank of J via matrix_rank (runs on CPU/LAPACK).

    normalize_rows=True (default): each row is divided by its L2 norm before
        SVD. Stabilises the threshold when rows span large magnitude ranges.
    normalize_rows=False: raw J is passed to matrix_rank. The default relative
        threshold (max(M,N)*eps*sigma_max) then reflects actual gradient magnitudes.
    """
    J_cpu = J.cpu()
    if normalize_rows:
        row_norms = torch.norm(J_cpu, dim=1, keepdim=True).clamp_min(1e-30)
        J_for_rank = J_cpu / row_norms
    else:
        J_for_rank = J_cpu
    if print_svd_info:
        svd_vals = torch.linalg.svdvals(J_for_rank)
        M, N = J_for_rank.shape
        sigma_max = svd_vals[0].item()
        eps = torch.finfo(J_for_rank.dtype).eps
        atol = max(M, N) * eps * sigma_max
        bottom = svd_vals[-10:].tolist()
        print(f"  [SVD] M={M} N={N}  normalised={normalize_rows}  "
              f"sigma_max={sigma_max:.3e}  atol={atol:.3e}  "
              f"bottom-10: {[f'{v:.3e}' for v in bottom]}", flush=True)
    return int(torch.linalg.matrix_rank(J_for_rank).item())


def compute_jacobian_rank(
    net,
    x_norm,
    y,
    criterion,
    keep_ids=None,
    entry_masks=None,
    max_entries=None,
    select_mode="topk_abs",
    device_for_J="cpu",
    print_svd_info=False,
):
    """Rank of J = d vec(g_obs) / d vec(x), built row-by-row to avoid OOM.

    print_svd_info: if True, print the 10 smallest singular values of the
                    row-normalised Jacobian to help diagnose rank behaviour.
    """
    J, total_entries, unknowns = _build_jacobian(
        net, x_norm, y, criterion, keep_ids, entry_masks, max_entries, select_mode, device_for_J,
    )
    jac_rank = _rank_of_J(J, print_svd_info=print_svd_info)
    return jac_rank, tuple(J.shape), J.shape[0], unknowns


def _qr_pivot_rows(J):
    """Return J reordered by QR column pivoting on J^T."""
    scipy_linalg = _load_scipy_linalg()
    if scipy_linalg is None:
        raise ImportError("qr_pivot requires scipy. Install with: pip install scipy")
    _, _, pivots = scipy_linalg.qr(J.cpu().numpy().T, pivoting=True, mode="economic")
    return J[torch.from_numpy(pivots.astype(np.int64))]


def compute_jacobian_rank_sweep(
    net,
    x_norm,
    y,
    criterion,
    keep_ids=None,
    entry_masks=None,
    row_counts=None,
    select_mode="topk_abs",
    qr_pivot=False,
    device_for_J="cpu",
    print_svd_info=False,
    j_progress_fn=None,
    rank_progress_fn=None,
    independent=True,
    normalize_rows=False,
    force_fc=False,
    fc_param_indices=None,
    print_layer_dist=False,
):
    """Rank of J for multiple row counts.

    independent=True (default):
        Builds J independently for each k using max_entries=k. Rank at k
        reflects exactly the information available when the attacker receives
        k gradient entries selected by select_mode. Theoretically correct for
        all select modes. Costs len(row_counts) × more forward passes.

    independent=False (pool-slice mode):
        Builds J once at max(row_counts) then slices J_max[:k] for each k.
        Fast (one J build per sample) but rank at k depends on max_row_count
        because the pool composition changes with the budget. Correct for
        topk_abs; misleading for layer_spread.

    qr_pivot in independent mode: J_k is already the full k-entry selection,
        so QR-pivoting it and taking all k rows gives the same rank as J_k.
        qr_pivot is most useful in pool-slice mode where it finds the best k
        rows from a larger pool.

    Returns ({k: (rank, shape, used_entries, unknowns)}, results_qr_or_None).
    """
    if not row_counts:
        raise ValueError("row_counts must be a non-empty list")

    if independent:
        results = {}
        results_qr = {} if qr_pivot else None
        for k in row_counts:
            J_k, _, unknowns = _build_jacobian(
                net, x_norm, y, criterion, keep_ids, entry_masks,
                k, select_mode, device_for_J, progress_fn=j_progress_fn,
                force_fc=force_fc, fc_param_indices=fc_param_indices,
                print_layer_dist=print_layer_dist,
            )
            rank = _rank_of_J(J_k, print_svd_info=print_svd_info, normalize_rows=normalize_rows)
            results[k] = (rank, tuple(J_k.shape), J_k.shape[0], unknowns)
            if rank_progress_fn is not None:
                rank_progress_fn(1)
            if qr_pivot:
                results_qr[k] = (rank, tuple(J_k.shape), J_k.shape[0], unknowns)
                if rank_progress_fn is not None:
                    rank_progress_fn(1)
            del J_k
        return results, results_qr

    # --- build-once (pool-slice) mode ---
    J_max, total_entries, unknowns = _build_jacobian(
        net, x_norm, y, criterion, keep_ids, entry_masks, max(row_counts), select_mode, device_for_J,
        progress_fn=j_progress_fn, force_fc=force_fc, fc_param_indices=fc_param_indices,
        print_layer_dist=print_layer_dist,
    )

    results = {}
    for k in row_counts:
        J_k = J_max[:k]
        rank = _rank_of_J(J_k, print_svd_info=print_svd_info, normalize_rows=normalize_rows)
        results[k] = (rank, tuple(J_k.shape), J_k.shape[0], unknowns)
        if rank_progress_fn is not None:
            rank_progress_fn(1)

    if not qr_pivot:
        return results, None

    J_max_qr = _qr_pivot_rows(J_max)
    results_qr = {}
    for k in row_counts:
        J_k = J_max_qr[:k]
        rank = _rank_of_J(J_k, print_svd_info=False, normalize_rows=normalize_rows)
        results_qr[k] = (rank, tuple(J_k.shape), J_k.shape[0], unknowns)
        if rank_progress_fn is not None:
            rank_progress_fn(1)

    return results, results_qr
