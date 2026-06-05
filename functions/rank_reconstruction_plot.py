"""
Visualise how reconstruction quality tracks Jacobian rank as the gradient
budget grows.

For each budget k the script:
  1. Computes Jacobian rank using compute_jacobian_rank_sweep — the exact same
     code path as jacobian_rank_sweep.py, so values match the table.
  2. Builds entry masks by mirroring _build_jacobian's selection logic
     (topk_abs or layer_spread), giving the attacker exactly k entries.
  3. Runs reconstruction with L-BFGS + L2 loss on those k entries.

Produces a figure:  GT | k=k1,rank=r1 | k=k2,rank=r2 | ...

Run example (LeNet / CIFAR-100, layer_spread):
  python3 -m functions.rank_reconstruction_plot --network lenet --dataset cifar100 --select_mode layer_spread --n_iter 300 --output rank_recon.png
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import transforms

import functions.consts as consts
from functions.Dataset import load_dataset
from functions.io_utils import resolve_storage_paths
from functions.masking import _get_last_fc_param_indices, flatten_observed_gradients, build_gradient_mask
from helper.metrics import (
    compute_grad_match_loss, total_variation,
    compute_jacobian_rank_sweep,
    _get_layer_groups, _layer_info,
)
from helper.Network import get_model, weights_init


# ---------------------------------------------------------------------------
# entry mask builder — mirrors _build_jacobian selection in helper/metrics.py
# ---------------------------------------------------------------------------

def _build_sweep_entry_masks(grads, keep_ids, budget, select_mode, net):
    """Build per-tensor entry_masks matching _build_jacobian's selection exactly.

    grads    : list of per-parameter gradients (from net.parameters() order)
    keep_ids : set of observed parameter indices (None = all)
    budget   : number of gradient entries to select
    select_mode: "topk_abs" or "layer_spread"
    net      : the network (needed for layer group info in layer_spread)

    Returns a list of boolean tensors (one per parameter, None if unobserved).
    """
    g_obs = flatten_observed_gradients(grads, keep_ids=keep_ids, entry_masks=None).detach()
    total_entries = g_obs.numel()
    k = min(int(budget), total_entries)

    if total_entries == 0:
        return [None] * len(grads)

    if k >= total_entries:
        selected_idx = torch.arange(total_entries, device=g_obs.device)
    elif select_mode == "topk_abs":
        selected_idx = torch.topk(g_obs.abs(), k=k, largest=True).indices
    elif select_mode == "layer_spread":
        # Mirror _build_jacobian layer_spread (metrics.py lines ~306–387):
        # Spread budget evenly across layer groups; within each group select top-|grad|.
        layer_sizes, _ = _layer_info(grads, keep_ids, None)
        named_params = list(net.named_parameters())
        param_to_group = _get_layer_groups(net)

        group_of = []
        for i, (name, _) in enumerate(named_params):
            if grads[i] is None:
                continue
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
                    indices.append(torch.arange(sz, device=g_obs.device) + off)
            else:
                grp_vals = torch.cat([g_obs[off:off + sz] for off, sz in slices])
                top_local = torch.topk(grp_vals.abs(), k=n, largest=True).indices
                local_to_global = torch.cat([
                    torch.arange(sz, device=g_obs.device) + off
                    for off, sz in slices
                ])
                indices.append(local_to_global[top_local])

        selected_idx = (torch.cat(indices) if indices
                        else torch.arange(k, device=g_obs.device))
    else:
        raise ValueError(f"Unknown select_mode: {select_mode!r}")

    # Convert flat selected_idx → per-tensor boolean masks
    flat_bool = torch.zeros(total_entries, dtype=torch.bool, device=g_obs.device)
    flat_bool[selected_idx] = True
    entry_masks = []
    offset = 0
    for i, g in enumerate(grads):
        if g is None:
            entry_masks.append(None)
            continue
        if keep_ids is not None and i not in keep_ids:
            entry_masks.append(None)
            continue
        n = g.numel()
        entry_masks.append(flat_bool[offset:offset + n].reshape(g.shape))
        offset += n
    return entry_masks


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------

def _print_mask_breakdown(entry_masks, named_params):
    """Print how many entries were selected from each parameter tensor."""
    rows = []
    for i, (name, _) in enumerate(named_params):
        m = entry_masks[i] if i < len(entry_masks) else None
        if m is None:
            continue
        kept = int(m.sum().item())
        if kept == 0:
            continue
        total = m.numel()
        rows.append((name, kept, total))
    for name, kept, total in rows:
        print(f"    {name}: {kept}/{total}")


# ---------------------------------------------------------------------------
# reconstruction
# ---------------------------------------------------------------------------

def _reconstruct(net, gt_data, gt_label, criterion, entry_masks,
                 dm, ds, lower_bound, upper_bound,
                 n_iter, lr, tv_weight, net_name_lower, device, seed):
    """L-BFGS + cosine loss reconstruction with the given entry mask (matches run_single_exp.py)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    all_params = list(net.parameters())
    selected_ids = [i for i, m in enumerate(entry_masks) if m is not None and m.any()]
    selected_params = [all_params[i] for i in selected_ids]

    # true gradients (detached)
    net.zero_grad()
    x_norm = (gt_data - dm) / ds
    out = net(x_norm)
    loss = criterion(out, gt_label)
    true_grads = torch.autograd.grad(loss, all_params)
    selected_original = [true_grads[i].detach() for i in selected_ids]
    selected_entry_masks = [entry_masks[i] for i in selected_ids]

    # label inference (iDLG rule: argmin sum of last-FC weight gradient)
    fc_ids = _get_last_fc_param_indices(net)
    named = list(net.named_parameters())
    fc_weight_idx = next(
        i for i in sorted(fc_ids) if named[i][0].endswith(".weight")
    )
    label_pred = torch.argmin(
        torch.sum(true_grads[fc_weight_idx], dim=-1), dim=-1
    ).detach().reshape((1,))

    dummy_data = torch.randn(gt_data.size(), device=device).requires_grad_(True)
    optimizer = torch.optim.LBFGS([dummy_data], lr=lr, max_iter=20, history_size=100)

    # LeNet uses sigmoid activations on raw [0,1] inputs — clamping between
    # L-BFGS steps corrupts the Hessian approximation and causes divergence.
    clamp_after_step = net_name_lower not in {"lenet", "lenet_bigger"}

    best_loss = float("inf")
    best_img = (dummy_data.detach() * ds + dm).clamp(0.0, 1.0)

    for _ in range(n_iter):
        def closure():
            optimizer.zero_grad()
            pred = net(dummy_data)
            dummy_loss = criterion(pred, label_pred)
            dummy_grads = torch.autograd.grad(dummy_loss, selected_params, create_graph=True)
            grad_diff, _ = compute_grad_match_loss(
                dummy_grads, selected_original,
                selected_entry_masks=selected_entry_masks, grad_loss="cos",
            )
            tv = total_variation(dummy_data)
            total = grad_diff + tv_weight * tv
            total.backward()
            return total

        optimizer.step(closure)

        if clamp_after_step:
            with torch.no_grad():
                dummy_data.clamp_(lower_bound, upper_bound)

        loss_val = closure().item()
        if np.isfinite(loss_val) and loss_val < best_loss:
            best_loss = loss_val
            current_x = (dummy_data.detach() * ds + dm).clamp(0.0, 1.0)
            best_img = current_x.clone()

        if loss_val < 1e-6:
            break

    return best_img


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--network",    default="lenet")
    parser.add_argument("--dataset",    default="cifar100")
    parser.add_argument("--sample_idx", type=int, default=35067,
                        help="Index into the dataset (deterministic).")
    parser.add_argument("--row_counts", default="3072,4000,5000,5500,6000,6500,7000",
                        help="Comma-separated gradient budgets.")
    parser.add_argument("--select_mode", default="topk_abs", choices=["topk_abs", "layer_spread"],
                        help="Entry selection strategy matching jacobian_rank_sweep.py.")
    parser.add_argument("--n_iter",     type=int, default=1000,
                        help="Reconstruction iterations (L-BFGS steps).")
    parser.add_argument("--lr",         type=float, default=1.0)
    parser.add_argument("--tv_weight",  type=float, default=0.0,
                        help="Total variation regularisation weight (default 0.0 matches main experiments).")
    parser.add_argument("--seed",       type=int, default=2)
    parser.add_argument("--device",     default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data_path",  default=None,
                        help="Override data directory (default: auto-resolved).")
    parser.add_argument("--output",     default="rank_recon.png")
    args = parser.parse_args()

    row_counts = [int(x) for x in args.row_counts.split(",")]
    device = torch.device(args.device)

    # Match jacobian_rank_sweep.py seed: run_id + 1 + worker_rank = 1 by default
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # get_model expects exact casing for custom nets
    _name_map = {"lenet": "LeNet", "lenet_bigger": "LeNet_bigger",
                 "mediumcnn": "MediumCNN", "biggercnn": "BiggerCNN"}
    network_name = _name_map.get(args.network.lower(), args.network)

    data_path = args.data_path or resolve_storage_paths(".")[0]
    dst, channel, num_classes, shape_img = load_dataset(args.dataset, data_path)

    net = get_model(network_name, channel=channel, num_classes=num_classes,
                    input_size=shape_img)
    net_name_lower = network_name.lower()
    if net_name_lower in {"lenet", "lenet_bigger"}:
        net.apply(weights_init)
    net = net.to(device).eval()

    criterion = nn.CrossEntropyLoss().to(device)

    # ---- normalisation (mirrors jacobian_rank_sweep.py) --------------------
    _sigmoid_nets = {"lenet", "lenet_bigger"}
    if net_name_lower in _sigmoid_nets:
        dm = torch.zeros(1, channel, 1, 1, device=device)
        ds = torch.ones(1, channel, 1, 1, device=device)
    else:
        dm = torch.tensor(getattr(consts, f"{args.dataset.lower()}_mean"),
                          device=device).view(1, channel, 1, 1)
        ds = torch.tensor(getattr(consts, f"{args.dataset.lower()}_std"),
                          device=device).view(1, channel, 1, 1)

    lower_bound = -dm / ds
    upper_bound = (1.0 - dm) / ds

    # ---- ground truth sample -----------------------------------------------
    gt_raw = transforms.ToTensor()(dst[args.sample_idx][0]).float().to(device).unsqueeze(0)
    gt_label = torch.tensor([dst[args.sample_idx][1]], dtype=torch.long, device=device)
    gt_norm = (gt_raw - dm) / ds
    gt_display = gt_raw.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1)

    # ---- compute all ranks upfront (sweep code path, float64) --------------
    # Convert to float64 before any gradient computation so that true_grads,
    # rank computation, and mask selection all use the same dtype —
    # matching _build_jacobian's internal dtype exactly.
    print(f"Computing Jacobian ranks ({args.select_mode}) for budgets {row_counts} ...")
    net.double()
    gt_norm_d = gt_norm.double()

    net.zero_grad()
    out = net(gt_norm_d)
    loss = criterion(out, gt_label)
    true_grads = [g.detach() for g in torch.autograd.grad(loss, net.parameters())]

    # keep_ids: method="idlg" returns all parameter indices observed
    keep_ids, _ = build_gradient_mask(
        method="idlg", mask_mode="gradsize_topfrac",
        net=net, original_dy_dx=true_grads,
    )

    rank_results, _ = compute_jacobian_rank_sweep(
        net=net,
        x_norm=gt_norm_d,
        y=gt_label,
        criterion=criterion,
        keep_ids=keep_ids,
        entry_masks=None,
        row_counts=row_counts,
        select_mode=args.select_mode,
        normalize_rows=False,
        independent=True,
        device_for_J="cpu",
    )
    net.float()

    # ---- per-budget reconstruction loop ------------------------------------
    named_params = list(net.named_parameters())
    results = []

    for k in row_counts:
        rank, _, _, unknowns = rank_results[k]
        print(f"\n=== budget = {k:,} ({args.select_mode}) ===")
        print(f"  rank = {rank} / {unknowns}")

        # Build masks using same float64 grads as _build_jacobian used internally
        entry_masks = _build_sweep_entry_masks(true_grads, keep_ids, k, args.select_mode, net)
        total_kept = sum(int(m.sum().item()) for m in entry_masks if m is not None)
        print(f"  total entries in mask: {total_kept}")
        print("  entries per layer:")
        _print_mask_breakdown(entry_masks, named_params)

        # Reconstruction
        print(f"  running reconstruction ({args.n_iter} iter) ...")
        recon = _reconstruct(
            net=net, gt_data=gt_raw, gt_label=gt_label, criterion=criterion,
            entry_masks=entry_masks, dm=dm, ds=ds,
            lower_bound=lower_bound, upper_bound=upper_bound,
            n_iter=args.n_iter, lr=args.lr,
            tv_weight=args.tv_weight, net_name_lower=net_name_lower,
            device=device, seed=args.seed,
        )
        recon_np = recon.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1)
        mse = float(np.mean((recon_np - gt_display) ** 2))
        psnr = -10 * np.log10(mse) if mse > 0 else float("inf")
        print(f"  MSE={mse:.4f}  PSNR={psnr:.2f} dB")

        results.append((f"entries = {k:,}", total_kept, rank, unknowns, recon_np))

    # ---- figure ------------------------------------------------------------
    n_panels = 1 + len(results)
    fig, axes = plt.subplots(1, n_panels, figsize=(3 * n_panels, 3.6))

    axes[0].imshow(gt_display)
    axes[0].set_title("Ground truth", fontsize=9, fontweight="bold")
    axes[0].axis("off")

    for ax, (label, total, rank, unknowns, img) in zip(axes[1:], results):
        ax.imshow(img)
        full = rank == unknowns
        rank_str = f"rank = {rank}/{unknowns}"
        mse = float(np.mean((img - gt_display) ** 2))
        psnr = -10 * np.log10(mse) if mse > 0 else float("inf")
        color = "#1a7f1a" if full else "#c0392b"
        ax.set_title(
            f"{label}\n{rank_str}\nPSNR = {psnr:.1f} dB",
            fontsize=8,
            color=color,
            fontweight="bold" if full else "normal",
        )
        ax.axis("off")

    fig.suptitle(
        f"{network_name} / {args.dataset} — reconstruction quality vs Jacobian rank  ({args.select_mode})",
        fontsize=9, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure to {args.output}")


if __name__ == "__main__":
    main()
