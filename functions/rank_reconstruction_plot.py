"""
Visualise how reconstruction quality tracks Jacobian rank as the gradient
budget grows.

For each budget k the script:
  1. Builds an entry mask  : global top-k non-FC entries by |grad|
                             (FC layer excluded from mask, J, and reconstruction;
                              used only for iDLG label inference)
                             (with --keep_fc: all last-FC entries + top-k non-FC)
  2. Runs reconstruction   : L-BFGS + L2 loss, 300 iterations, lr=1
  3. Computes Jacobian rank: with that exact entry mask

Produces a figure:  GT | k=k1,rank=r1 | k=k2,rank=r2 | ...

Run example (LeNet / CIFAR-100, no keep_fc):
  python3 -m functions.rank_reconstruction_plot --network lenet --dataset cifar100 --row_counts 3072,4000,5000,6000 --sample_idx 0 --n_iter 300 --output rank_recon.png

With keep_fc (row_counts = non-FC budget on top of FC):
  python3 -m functions.rank_reconstruction_plot --network lenet --dataset cifar100 --row_counts 3072,4000,5000,6000 --keep_fc --sample_idx 0 --n_iter 300 --output rank_recon_keepfc.png
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
from functions.masking import _get_last_fc_param_indices
from helper.metrics import compute_grad_match_loss, total_variation, compute_jacobian_rank
from helper.Network import get_model, weights_init

# ---------------------------------------------------------------------------
# mask helpers
# ---------------------------------------------------------------------------

def _build_global_topk_masks(grads, fc_ids, budget):
    """Global top-k entries by |grad| across non-FC parameters only. FC params excluded entirely."""
    all_info = []
    for i, g in enumerate(grads):
        if g is None or i in fc_ids:
            continue
        all_info.append((i, g.shape, g.detach().abs().reshape(-1)))

    entry_masks = [None] * len(grads)  # FC stays None → excluded from J and reconstruction

    if all_info and budget > 0:
        all_vals = torch.cat([v for _, _, v in all_info])
        k = min(int(budget), all_vals.numel())
        top_idx = torch.topk(all_vals, k=k, largest=True).indices
        gmask = torch.zeros(all_vals.numel(), dtype=torch.bool, device=all_vals.device)
        gmask[top_idx] = True
        offset = 0
        for i, shape, flat in all_info:
            n = flat.numel()
            entry_masks[i] = gmask[offset:offset + n].reshape(shape)
            offset += n

    return entry_masks


def _build_keepfc_masks(grads, fc_ids, non_fc_budget):
    """FC params: all entries True. Non-FC: global top-k by |grad| True."""
    non_fc_info = []
    for i, g in enumerate(grads):
        if g is None or i in fc_ids:
            continue
        non_fc_info.append((i, g.shape, g.detach().abs().reshape(-1)))

    entry_masks = [None] * len(grads)

    if non_fc_info and non_fc_budget > 0:
        all_vals = torch.cat([v for _, _, v in non_fc_info])
        k = min(int(non_fc_budget), all_vals.numel())
        top_idx = torch.topk(all_vals, k=k, largest=True).indices
        gmask = torch.zeros(all_vals.numel(), dtype=torch.bool, device=all_vals.device)
        gmask[top_idx] = True
        offset = 0
        for i, shape, flat in non_fc_info:
            n = flat.numel()
            entry_masks[i] = gmask[offset:offset + n].reshape(shape)
            offset += n

    for i, g in enumerate(grads):
        if g is not None and i in fc_ids:
            entry_masks[i] = torch.ones(g.shape, dtype=torch.bool, device=g.device)

    return entry_masks


# ---------------------------------------------------------------------------
# reconstruction
# ---------------------------------------------------------------------------

def _reconstruct(net, gt_data, gt_label, criterion, entry_masks,
                 dm, ds, lower_bound, upper_bound,
                 n_iter, lr, tv_weight, net_name_lower, device, seed):
    """L-BFGS + L2 loss reconstruction with the given entry mask."""
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
                selected_entry_masks=selected_entry_masks, grad_loss="l2",
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

    return best_img


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--network",    default="lenet")
    parser.add_argument("--dataset",    default="cifar100")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Index into the dataset (deterministic).")
    parser.add_argument("--row_counts", default="3072,4000,5000,6000",
                        help="Comma-separated gradient budgets (total entries without --keep_fc; non-FC budget with --keep_fc).")
    parser.add_argument("--keep_fc",    action="store_true",
                        help="Force-keep all last-FC entries; row_counts then refers to non-FC budget.")
    parser.add_argument("--n_iter",     type=int, default=300,
                        help="Reconstruction iterations (L-BFGS steps).")
    parser.add_argument("--lr",         type=float, default=1.0)
    parser.add_argument("--tv_weight",  type=float, default=0.0,
                        help="Total variation regularisation weight (default 0.0 matches main experiments).")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--device",     default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data_path",  default=None,
                        help="Override data directory (default: auto-resolved).")
    parser.add_argument("--output",     default="rank_recon.png")
    args = parser.parse_args()

    row_counts = [int(x) for x in args.row_counts.split(",")]
    device = torch.device(args.device)

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

    # ---- normalisation (mirrors run_single_exp.py) -------------------------
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

    # ---- true gradients once (masks are built from these) ------------------
    net.zero_grad()
    out = net(gt_norm)
    loss = criterion(out, gt_label)
    true_grads = [g.detach() for g in torch.autograd.grad(loss, net.parameters())]

    fc_ids = _get_last_fc_param_indices(net)
    fc_size = sum(true_grads[i].numel() for i in fc_ids)
    if args.keep_fc:
        print(f"FC entries (force-kept): {fc_size}")

    # ---- per-budget loop ---------------------------------------------------
    results = []

    for k in row_counts:
        if args.keep_fc:
            print(f"\n=== non-FC budget = {k} ===")
            entry_masks = _build_keepfc_masks(true_grads, fc_ids, k)
            total_kept = sum(int(m.sum().item()) for m in entry_masks if m is not None)
            print(f"  total entries in mask (non-FC={k} + FC={fc_size}): {total_kept}")
            budget_label = f"non-FC = {k:,}"
        else:
            print(f"\n=== total budget = {k} ===")
            entry_masks = _build_global_topk_masks(true_grads, fc_ids, k)
            total_kept = sum(int(m.sum().item()) for m in entry_masks if m is not None)
            print(f"  total entries in mask: {total_kept}")
            budget_label = f"entries = {k:,}"

        # Jacobian rank — use float64 to avoid rank underestimation from float32 eps
        print(f"  computing Jacobian rank ...")
        net.double()
        rank, _, n_rows, unknowns = compute_jacobian_rank(
            net=net, x_norm=gt_norm.double(), y=gt_label, criterion=criterion,
            keep_ids=None, entry_masks=entry_masks,
            max_entries=None, select_mode="topk_abs", device_for_J="cpu",
        )
        net.float()
        print(f"  rank = {rank} / {unknowns}")

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

        results.append((budget_label, total_kept, rank, unknowns, recon_np))

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

    mode_str = "keep_fc" if args.keep_fc else "global topk"
    fig.suptitle(
        f"{network_name} / {args.dataset} — reconstruction quality vs Jacobian rank  ({mode_str})",
        fontsize=9, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure to {args.output}")


if __name__ == "__main__":
    main()
