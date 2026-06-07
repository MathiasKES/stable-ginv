"""
Visualise how reconstruction quality tracks Jacobian rank as the gradient
budget grows.

Three selection modes (--select_mode):
  topk_abs                       : global top-k entries by |grad|
  layer_spread                   : budget spread evenly across layer groups
  gradsize_topfrac_entries_layer : per-layer top-frac (matches masking experiments)

Add --exclude_fc to remove the last FC layer from the entry pool in all modes.
FC is always kept for iDLG label inference regardless of this flag.

Run examples:
  # topk_abs, FC excluded — clean rank->reconstruction figure
  python3 -m stable_ginv.viz.plot_rank_reconstruction --select_mode topk_abs --exclude_fc --row_counts 3072,4000,5000,5500,6000,7000 --sample_idx 12196 --seed 1 --output rank_recon_topk_nofc.png

  # topk_abs, FC in pool — rank not sufficient figure
  python3 -m stable_ginv.viz.plot_rank_reconstruction --select_mode topk_abs --row_counts 3072,5000,6000,7000 --sample_idx 12196 --seed 1 --output rank_recon_topk_fc.png

  # gradsize_topfrac_entries_layer, FC excluded — matches masking experiments
  python3 -m stable_ginv.viz.plot_rank_reconstruction --select_mode gradsize_topfrac_entries_layer --exclude_fc --topfrac_values 1.0,0.7,0.5,0.3,0.1 --sample_idx 12196 --seed 1 --output rank_recon_topfrac_nofc.png
"""

import argparse
import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import transforms

import functions.consts as consts
from functions.Dataset import load_dataset
from stable_ginv.io import resolve_storage_paths
from stable_ginv.masking import _get_last_fc_param_indices, build_gradient_mask
from stable_ginv.metrics import (
    compute_grad_match_loss, total_variation,
    compute_jacobian_rank,
    _get_layer_groups,
)
from helper.Network import get_model, weights_init
from stable_ginv.recon import make_scheduler


# ---------------------------------------------------------------------------
# pool worker — module-level so multiprocessing.Pool can pickle it
# ---------------------------------------------------------------------------

def _recon_worker(task):
    """Reconstruct one budget on CPU in a worker process; return metrics + image.

    ``task`` is a flat picklable tuple (masks/tensors as numpy) so it crosses the
    multiprocessing boundary. Returns ``(task_idx, budget_label, total_kept, rank,
    unknowns, recon_np)``.
    """
    (task_idx, budget_label, rank, unknowns, masks_np,
     net_state, network_name, channel, num_classes, shape_img,
     gt_raw_np, gt_label_val, dm_np, ds_np, lb_np, ub_np,
     n_iter, lr, tv_weight, net_name_lower, seed, optimizer_name) = task

    device = torch.device("cpu")
    entry_masks = [torch.from_numpy(m).bool() if m is not None else None for m in masks_np]
    gt_raw      = torch.from_numpy(gt_raw_np)
    gt_label    = torch.tensor([gt_label_val], dtype=torch.long)
    dm          = torch.from_numpy(dm_np)
    ds          = torch.from_numpy(ds_np)
    lower_bound = torch.from_numpy(lb_np)
    upper_bound = torch.from_numpy(ub_np)

    net = get_model(network_name, channel=channel, num_classes=num_classes, input_size=shape_img)
    net.load_state_dict(net_state)
    net = net.to(device).eval()
    criterion = nn.CrossEntropyLoss()

    total_kept = sum(int(m.sum().item()) for m in entry_masks if m is not None)
    recon = _reconstruct(
        net=net, gt_data=gt_raw, gt_label=gt_label, criterion=criterion,
        entry_masks=entry_masks, dm=dm, ds=ds,
        lower_bound=lower_bound, upper_bound=upper_bound,
        n_iter=n_iter, lr=lr, tv_weight=tv_weight,
        net_name_lower=net_name_lower, device=device, seed=seed,
        optimizer_name=optimizer_name,
    )
    recon_np = recon.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1)
    return task_idx, budget_label, total_kept, rank, unknowns, recon_np


# ---------------------------------------------------------------------------
# entry mask builders
# ---------------------------------------------------------------------------

def _build_entry_masks(grads, fc_ids, budget, select_mode, net, exclude_fc):
    """Build per-tensor boolean masks for topk_abs or layer_spread modes."""
    obs_info = [
        (i, g) for i, g in enumerate(grads)
        if g is not None and (not exclude_fc or i not in fc_ids)
    ]
    if not obs_info:
        return [None] * len(grads)

    total = sum(g.numel() for _, g in obs_info)
    k = min(int(budget), total)

    if select_mode == "topk_abs":
        all_flat = torch.cat([g.detach().abs().reshape(-1) for _, g in obs_info])
        sel = torch.zeros(total, dtype=torch.bool, device=all_flat.device)
        if k >= total:
            sel[:] = True
        else:
            sel[torch.topk(all_flat, k=k, largest=True).indices] = True

    elif select_mode == "layer_spread":
        named_params = list(net.named_parameters())
        param_to_group = _get_layer_groups(net)

        group_of = [
            param_to_group.get(named_params[i][0], named_params[i][0].split(".")[0])
            for i, _ in obs_info
        ]
        seen, groups = set(), []
        for grp in group_of:
            if grp not in seen:
                groups.append(grp); seen.add(grp)

        layer_sizes = [g.numel() for _, g in obs_info]
        group_totals = {grp: 0 for grp in groups}
        group_slices = {grp: [] for grp in groups}
        off = 0
        for sz, grp in zip(layer_sizes, group_of):
            group_totals[grp] += sz
            group_slices[grp].append((off, sz))
            off += sz

        num_g = len(groups)
        base, rem = k // num_g, k % num_g
        order = sorted(groups, key=lambda g: group_totals[g], reverse=True)
        per_group = {grp: min(base + (1 if j < rem else 0), group_totals[grp])
                     for j, grp in enumerate(order)}
        leftover = k - sum(per_group.values())
        for grp in order:
            if leftover <= 0: break
            cap = group_totals[grp] - per_group[grp]
            if cap > 0:
                give = min(leftover, cap)
                per_group[grp] += give; leftover -= give

        all_flat = torch.cat([g.detach().abs().reshape(-1) for _, g in obs_info])
        indices = []
        for grp in groups:
            n, slices = per_group[grp], group_slices[grp]
            if n <= 0: continue
            if n >= group_totals[grp]:
                for o, sz in slices:
                    indices.append(torch.arange(sz, device=all_flat.device) + o)
            else:
                grp_flat = torch.cat([all_flat[o:o+sz] for o, sz in slices])
                top_local = torch.topk(grp_flat, k=n, largest=True).indices
                g2g = torch.cat([torch.arange(sz, device=all_flat.device) + o for o, sz in slices])
                indices.append(g2g[top_local])

        sel = torch.zeros(total, dtype=torch.bool, device=all_flat.device)
        if indices:
            sel[torch.cat(indices)] = True

    else:
        raise ValueError(f"Unknown select_mode: {select_mode!r}")

    entry_masks = [None] * len(grads)
    off = 0
    for i, g in obs_info:
        n = g.numel()
        entry_masks[i] = sel[off:off+n].reshape(g.shape)
        off += n
    return entry_masks


def _build_topfrac_masks(grads, fc_ids, topfrac, net, exclude_fc):
    """Per-layer top-frac masks matching gradsize_topfrac_entries_layer experiments."""
    _, entry_masks = build_gradient_mask(
        method="masked",
        mask_mode="gradsize_topfrac_entries_layer",
        net=net,
        original_dy_dx=grads,
        gradsize_topfrac=topfrac,
    )
    if exclude_fc and entry_masks is not None:
        for i in fc_ids:
            if i < len(entry_masks):
                entry_masks[i] = None
    return entry_masks


# ---------------------------------------------------------------------------
# reconstruction
# ---------------------------------------------------------------------------

def _reconstruct(net, gt_data, gt_label, criterion, entry_masks,
                 dm, ds, lower_bound, upper_bound,
                 n_iter, lr, tv_weight, net_name_lower, device, seed,
                 optimizer_name="lbfgs", gamma=0.5):
    """Reconstruct the input for one entry-mask budget; return the best image.

    Optimizes dummy pixels to match the masked observed gradients (L-BFGS + L2,
    or Adam/AdamW + cosine with a scheduler), keeping the lowest-loss iterate.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    all_params = list(net.parameters())
    selected_ids = [i for i, m in enumerate(entry_masks) if m is not None and m.any()]
    selected_params = [all_params[i] for i in selected_ids]

    net.zero_grad()
    x_norm = (gt_data - dm) / ds
    out = net(x_norm)
    loss = criterion(out, gt_label)
    true_grads = torch.autograd.grad(loss, all_params)
    selected_original = [true_grads[i].detach() for i in selected_ids]
    selected_entry_masks = [entry_masks[i] for i in selected_ids]

    fc_ids = _get_last_fc_param_indices(net)
    named = list(net.named_parameters())
    fc_weight_idx = next(i for i in sorted(fc_ids) if named[i][0].endswith(".weight"))
    label_pred = torch.argmin(
        torch.sum(true_grads[fc_weight_idx], dim=-1), dim=-1
    ).detach().reshape((1,))

    dummy_data = torch.randn(gt_data.size(), device=device).requires_grad_(True)

    grad_loss = "l2" if optimizer_name == "lbfgs" else "cos"
    scheduler = None

    if optimizer_name == "lbfgs":
        optimizer = torch.optim.LBFGS([dummy_data], lr=lr, max_iter=20, history_size=100)
        phase = "lbfgs"
    elif optimizer_name in ("adam", "signed_adam"):
        optimizer = torch.optim.Adam([dummy_data], lr=lr)
        scheduler = make_scheduler(optimizer, n_iter, gamma=gamma)
        phase = optimizer_name
    elif optimizer_name in ("adamw", "signed_adamw"):
        optimizer = torch.optim.AdamW([dummy_data], lr=lr, weight_decay=1e-5)
        scheduler = make_scheduler(optimizer, n_iter, gamma=gamma)
        phase = optimizer_name
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name!r}")

    best_loss = float("inf")
    best_img  = (dummy_data.detach() * ds + dm).clamp(0.0, 1.0)

    for _ in range(n_iter):
        if phase == "lbfgs":
            def closure():
                """L-BFGS closure: gradient-matching + TV loss, with backprop."""
                optimizer.zero_grad()
                pred = net(dummy_data)
                dummy_loss = criterion(pred, label_pred)
                dummy_grads = torch.autograd.grad(dummy_loss, selected_params, create_graph=True)
                grad_diff, _ = compute_grad_match_loss(
                    dummy_grads, selected_original,
                    selected_entry_masks=selected_entry_masks, grad_loss=grad_loss,
                )
                total = grad_diff + tv_weight * total_variation(dummy_data)
                total.backward()
                return total

            optimizer.step(closure)
            # LeNet sigmoid inputs — no clamp after LBFGS (corrupts quasi-Newton approx)
            if net_name_lower not in {"lenet", "lenet_bigger"}:
                with torch.no_grad():
                    dummy_data.clamp_(lower_bound, upper_bound)
            current_loss = closure().item()

        else:
            optimizer.zero_grad()
            pred = net(dummy_data)
            dummy_loss = criterion(pred, label_pred)
            dummy_grads = torch.autograd.grad(dummy_loss, selected_params, create_graph=True)
            grad_diff, _ = compute_grad_match_loss(
                dummy_grads, selected_original,
                selected_entry_masks=selected_entry_masks, grad_loss=grad_loss,
            )
            total_loss = grad_diff + tv_weight * total_variation(dummy_data)
            total_loss.backward()
            if phase in ("signed_adam", "signed_adamw") and dummy_data.grad is not None:
                with torch.no_grad():
                    dummy_data.grad.sign_()
            optimizer.step()
            with torch.no_grad():
                dummy_data.clamp_(lower_bound, upper_bound)
            if scheduler is not None:
                scheduler.step()
            current_loss = total_loss.item()

        if np.isfinite(current_loss) and current_loss < best_loss:
            best_loss = current_loss
            best_img  = (dummy_data.detach() * ds + dm).clamp(0.0, 1.0).clone()

        if current_loss < 1e-6:
            break

    return best_img


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    """Run the rank-vs-reconstruction sweep and save the comparison figure.

    Parses CLI arguments, builds an entry mask per budget (``--row_counts`` for
    topk_abs/layer_spread, or ``--topfrac_values`` for
    gradsize_topfrac_entries_layer), reconstructs each budget (optionally across
    worker processes), computes the Jacobian rank for each, and writes a
    ground-truth-plus-budgets panel figure to ``--output``.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--network",    default="lenet")
    parser.add_argument("--dataset",    default="cifar100")
    parser.add_argument("--sample_idx", type=int, default=12196)
    parser.add_argument("--select_mode", default="topk_abs",
                        choices=["topk_abs", "layer_spread", "gradsize_topfrac_entries_layer"])
    parser.add_argument("--exclude_fc", action="store_true",
                        help="Exclude last FC layer from entry pool (still used for label inference).")
    # topk_abs / layer_spread axes
    parser.add_argument("--row_counts", default="3072,4000,5000,5500,6000,7000",
                        help="Comma-separated entry budgets (topk_abs / layer_spread modes).")
    # gradsize_topfrac_entries_layer axis
    parser.add_argument("--topfrac_values", default="1.0,0.7,0.5,0.3,0.1",
                        help="Comma-separated topfrac values (gradsize_topfrac_entries_layer mode).")
    parser.add_argument("--optimizer",  default="lbfgs",
                        choices=["lbfgs", "signed_adamw", "adamw", "signed_adam", "adam"])
    parser.add_argument("--n_iter",     type=int, default=300,
                        help="Reconstruction iterations (300 for lbfgs, 5000 for signed_adamw).")
    parser.add_argument("--lr",         type=float, default=1.0)
    parser.add_argument("--tv_weight",  type=float, default=0.0)
    parser.add_argument("--seed",       type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--device",     default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data_path",  default=None)
    parser.add_argument("--output",     default="rank_recon.png")
    parser.add_argument("--labels_position", default="above", choices=["above", "below"],
                        help="Place panel labels above or below each image.")
    parser.add_argument("--label_fontsize", type=int, default=20,
                        help="Font size for panel labels.")
    args = parser.parse_args()

    use_topfrac = (args.select_mode == "gradsize_topfrac_entries_layer")
    budgets = (
        [float(x) for x in args.topfrac_values.split(",")]
        if use_topfrac else
        [int(x) for x in args.row_counts.split(",")]
    )

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    _name_map = {"lenet": "LeNet", "lenet_bigger": "LeNet_bigger",
                 "mediumcnn": "MediumCNN", "biggercnn": "BiggerCNN"}
    network_name = _name_map.get(args.network.lower(), args.network)
    net_name_lower = network_name.lower()

    data_path = args.data_path or resolve_storage_paths(".")[0]
    dst, channel, num_classes, shape_img = load_dataset(args.dataset, data_path)

    net = get_model(network_name, channel=channel, num_classes=num_classes, input_size=shape_img)
    if net_name_lower in {"lenet", "lenet_bigger"}:
        net.apply(weights_init)
    net = net.to(device).eval()

    criterion = nn.CrossEntropyLoss().to(device)

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

    gt_raw   = transforms.ToTensor()(dst[args.sample_idx][0]).float().to(device).unsqueeze(0)
    gt_label = torch.tensor([dst[args.sample_idx][1]], dtype=torch.long, device=device)
    gt_norm  = (gt_raw - dm) / ds
    gt_display = gt_raw.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1)

    # float32 gradients for mask building
    net.zero_grad()
    out  = net(gt_norm)
    loss = criterion(out, gt_label)
    true_grads_f32 = [g.detach() for g in torch.autograd.grad(loss, net.parameters())]
    fc_ids = _get_last_fc_param_indices(net)

    fc_str = "FC excluded" if args.exclude_fc else "FC in pool"
    print(f"Mode: {args.select_mode}  |  {fc_str}  |  sample {args.sample_idx}")

    # ---- build masks, compute rank (float64), assemble tasks ----------------
    net_state = {k: v.cpu() for k, v in net.state_dict().items()}
    tasks = []

    for task_idx, budget in enumerate(budgets):
        if use_topfrac:
            entry_masks  = _build_topfrac_masks(true_grads_f32, fc_ids, budget, net, args.exclude_fc)
            budget_label = f"topfrac = {budget:.1f}"
        else:
            entry_masks  = _build_entry_masks(true_grads_f32, fc_ids, budget,
                                              args.select_mode, net, args.exclude_fc)
            budget_label = f"entries = {int(budget):,}"

        total_kept = sum(int(m.sum().item()) for m in entry_masks if m is not None)

        # rank in float64 to avoid underestimation
        net.double()
        rank, _, _, unknowns = compute_jacobian_rank(
            net=net, x_norm=gt_norm.double(), y=gt_label, criterion=criterion,
            keep_ids=None, entry_masks=entry_masks,
            max_entries=None, select_mode="topk_abs", device_for_J="cpu",
        )
        net.float()

        print(f"  {budget_label}  entries kept={total_kept}  rank={rank}/{unknowns}")

        masks_np = [m.cpu().numpy() if m is not None else None for m in entry_masks]
        tasks.append((
            task_idx, budget_label, rank, unknowns, masks_np,
            net_state, network_name, channel, num_classes, shape_img,
            gt_raw.cpu().numpy(), gt_label.item(),
            dm.cpu().numpy(), ds.cpu().numpy(),
            lower_bound.cpu().numpy(), upper_bound.cpu().numpy(),
            args.n_iter, args.lr, args.tv_weight, net_name_lower, args.seed,
            args.optimizer,
        ))

    # ---- reconstruct --------------------------------------------------------
    n_workers = min(args.num_workers, len(tasks))
    print(f"\nRunning {len(tasks)} reconstructions ({args.n_iter} iter, {n_workers} worker(s)) ...")
    if n_workers > 1:
        with mp.get_context("spawn").Pool(n_workers) as pool:
            raw = pool.map(_recon_worker, tasks)
        raw.sort(key=lambda x: x[0])
    else:
        raw = [_recon_worker(t) for t in tasks]

    results = []
    for _, label, total, rank, unknowns, recon_np in raw:
        mse  = float(np.mean((recon_np - gt_display) ** 2))
        psnr = -10 * np.log10(mse) if mse > 0 else float("inf")
        print(f"  {label}  rank={rank}/{unknowns}  MSE={mse:.4f}  PSNR={psnr:.2f} dB")
        results.append((label, rank, unknowns, recon_np))

    # ---- figure -------------------------------------------------------------
    n_panels = 1 + len(results)
    fs       = args.label_fontsize
    above    = (args.labels_position == "above")

    # Figure height chosen so the axes are square (matching the square CIFAR
    # images) and the label rows still fit.  With left=0.005, right=0.995,
    # wspace=0.02 and n_panels panels, each panel is ~2.92 inches wide, so we
    # need axes_height = 2.92 inches = 0.68 × 4.3 inches.
    fig, axes = plt.subplots(1, n_panels, figsize=(3 * n_panels, 4.3))

    def _set_label(ax, text, color, bold):
        """Place a panel label as a title (above) or caption (below)."""
        kw = dict(fontsize=fs, color=color,
                  fontweight="bold" if bold else "normal",
                  ha="center")
        if above:
            ax.set_title(text, **kw)
        else:
            ax.text(0.5, -0.01, text, va="top", transform=ax.transAxes, **kw)

    axes[0].imshow(gt_display)
    _set_label(axes[0], "Ground truth", "black", True)
    axes[0].axis("off")

    for ax, (label, rank, unknowns, img) in zip(axes[1:], results):
        ax.imshow(img)
        full  = rank == unknowns
        mse   = float(np.mean((img - gt_display) ** 2))
        psnr  = -10 * np.log10(mse) if mse > 0 else float("inf")
        color = "#1a7f1a" if full else "#c0392b"
        text  = f"{label}\nrank = {rank}/{unknowns}\nPSNR = {psnr:.1f} dB"
        _set_label(ax, text, color, full)
        ax.axis("off")

    # Axes occupy 68% of figure height (bottom=0.02 to top=0.70), making them
    # square to match the square CIFAR images.  The remaining 30% is reserved
    # for label text above (FC) or below (noFC).  Both orientations produce
    # identical figure dimensions so they stack flush in the document.
    if above:
        plt.subplots_adjust(top=0.70, bottom=0.02, left=0.005, right=0.995, wspace=0.02)
    else:
        plt.subplots_adjust(top=0.98, bottom=0.30, left=0.005, right=0.995, wspace=0.02)

    plt.savefig(args.output, dpi=150)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
