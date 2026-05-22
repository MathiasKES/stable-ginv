import os
import csv
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import consts
from Dataset import lfw_dataset
from Network import weights_init
from Misc_functions import (
    build_network,
    build_gradient_mask,
    compute_jacobian_rank,
)


def load_dataset(dataset, data_path):
    if dataset == 'MNIST':
        shape_img = (28, 28)
        num_classes = 10
        channel = 1
        dst = datasets.MNIST(data_path, download=True)
    elif dataset == 'cifar100':
        shape_img = (32, 32)
        num_classes = 100
        channel = 3
        dst = datasets.CIFAR100(data_path, download=True)
    elif dataset == 'cifar10':
        shape_img = (32, 32)
        num_classes = 10
        channel = 3
        dst = datasets.CIFAR10(data_path, download=True)
    elif dataset == 'lfw':
        shape_img = (32, 32)
        num_classes = 5749
        channel = 3
        lfw_path = os.path.join(data_path, 'lfw')
        os.makedirs(lfw_path, exist_ok=True)
        dst = lfw_dataset(lfw_path, shape_img)
    else:
        raise ValueError('unknown dataset')
    return dst, channel, num_classes, shape_img


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--network", type=str, default="resnet18")
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--method", type=str, default="idlg", choices=["idlg", "masked"])

    parser.add_argument("--mask_mode", type=str, default="gradsize_topfrac")
    parser.add_argument("--prefixes", type=str, default="conv1,layer1,layer2,layer3,fc")
    parser.add_argument("--gradsize_topk", type=int, default=20)
    parser.add_argument("--gradsize_topfrac", type=float, default=0.)
    parser.add_argument("--gradsize_threshold", type=float, default=None)
    parser.add_argument("--gradsize_metric", type=str, default="l2")

    parser.add_argument("--row_counts", type=str, default="3072,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000")
    parser.add_argument("--jacobian_select_mode", type=str, default="topk_abs",
                        choices=["topk_abs", "first", "random"])
    parser.add_argument("--rank_tol", type=float, default=1e-6,
                    help="Relative tolerance for numerical Jacobian rank.")
    parser.add_argument("--normalize_jacobian_rows", action="store_true",
                    help="If set, normalize Jacobian rows before computing singular values.")

    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")


    args = parser.parse_args()

    row_counts = [int(x.strip()) for x in args.row_counts.split(",") if x.strip()]
    #prefixes = tuple(p.strip() for p in args.prefixes.split(",") if p.strip())
    prefixes_list = []
    prefix_layer_fracs = {}

    for item in args.prefixes.split(","):
        item = item.strip()
        if not item:
            continue

        if ":" in item:
            prefix, frac = item.split(":", 1)
            prefix = prefix.strip()
            frac = float(frac.strip())

            prefixes_list.append(prefix)
            prefix_layer_fracs[prefix] = frac
        else:
            prefixes_list.append(item)

    prefixes = tuple(prefixes_list)

    if os.access('/work3/s234843/bachelor', os.R_OK | os.W_OK | os.X_OK):
        data_path = '/work3/s234843/bachelor/datasets'
        save_dir = '/work3/s234843/bachelor/results'
    else:
        data_path = './data'
        save_dir = './results'

    os.makedirs(save_dir, exist_ok=True)

    dst, channel, num_classes, shape_img = load_dataset(args.dataset, data_path)

    seed = args.run_id + 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    device = args.device
    net = build_network(args.network, channel=channel, num_classes=num_classes, input_size=shape_img)
    if not args.network.startswith("resnet"):
        net.apply(weights_init)
    net = net.to(device).double()
    net.eval()

    tt = transforms.Compose([transforms.ToTensor()])
    criterion = nn.CrossEntropyLoss().to(device)

    if channel == 1:
        dm = torch.tensor(getattr(consts, f'{args.dataset.lower()}_mean'), device=device, dtype=torch.float64).view(1,1,1,1)
        ds = torch.tensor(getattr(consts, f'{args.dataset.lower()}_std'), device=device, dtype=torch.float64).view(1,1,1,1)
    else:
        dm = torch.tensor(getattr(consts, f'{args.dataset.lower()}_mean'), device=device, dtype=torch.float64).view(1,channel,1,1)
        ds = torch.tensor(getattr(consts, f'{args.dataset.lower()}_std'), device=device, dtype=torch.float64).view(1,channel,1,1)

    idx_shuffle = np.random.permutation(len(dst))
    sample_indices = idx_shuffle[:args.num_samples]

    unknowns = channel * shape_img[0] * shape_img[1]
    rank_results = {rows: [] for rows in row_counts}

    print(f"Unknowns: {unknowns}")
    print(f"Samples: {sample_indices.tolist()}")

    for s_idx, idx in enumerate(sample_indices):
        print(f"\nSample {s_idx+1}/{args.num_samples}, dataset idx={idx}", flush=True)

        gt_data = tt(dst[idx][0]).double().to(device).unsqueeze(0)
        gt_label = torch.tensor([dst[idx][1]], dtype=torch.long, device=device)

        gt_data_norm = (gt_data - dm) / ds
        out = net(gt_data_norm)
        loss = criterion(out, gt_label)
        dy_dx = torch.autograd.grad(loss, net.parameters())
        original_dy_dx = [g.detach().clone() for g in dy_dx]

        keep_ids, entry_masks = build_gradient_mask(
            method=args.method,
            mask_mode=args.mask_mode,
            net=net,
            original_dy_dx=original_dy_dx,
            prefixes=prefixes,
            prefix_layer_fracs=prefix_layer_fracs,
            gradsize_topk=args.gradsize_topk,
            gradsize_topfrac=args.gradsize_topfrac,
            gradsize_threshold=args.gradsize_threshold,
            gradsize_metric=args.gradsize_metric,
        )

        for rows in row_counts:
            jac_rank, jac_shape, used_rows, jac_unknowns = compute_jacobian_rank(
                net=net,
                x_norm=gt_data_norm,
                y=gt_label,
                criterion=criterion,
                keep_ids=keep_ids,
                entry_masks=entry_masks,
                max_entries=rows,
                select_mode=args.jacobian_select_mode,
                device_for_J="cpu",
                rank_tol=args.rank_tol,
                normalize_rows=args.normalize_jacobian_rows,
            )

            print(f"  rows={rows}, rank={jac_rank}, shape={jac_shape}", flush=True)
            rank_results[rows].append(jac_rank)

    xs = []
    mean_ranks = []
    std_ranks = []

    for rows in row_counts:
        xs.append(rows)
        mean_ranks.append(float(np.mean(rank_results[rows])))
        std_ranks.append(float(np.std(rank_results[rows])))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = (
    f"jac_rank_{args.network}_{args.dataset}_{args.method}_"
    f"{args.jacobian_select_mode}_ns{args.num_samples}_{timestamp}"
    )
    csv_path = os.path.join(save_dir, base + ".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rows_used",
            "mean_rank",
            "std_rank",
            "unknowns",
            "num_samples",
            "jacobian_select_mode",
        ])
        for x, m, s in zip(xs, mean_ranks, std_ranks):
            writer.writerow([
                x,
                m,
                s,
                unknowns,
                args.num_samples,
                args.jacobian_select_mode,
            ])

    legend_label = (
        f"mean rank "
        f"(select={args.jacobian_select_mode}, samples={args.num_samples}"
    )

    if args.method == "masked":
        legend_label += f", mask={args.mask_mode}"

    legend_label += ")"

    plt.figure(figsize=(7, 5))
    plt.errorbar(
        xs,
        mean_ranks,
        yerr=std_ranks,
        marker="o",
        capsize=4,
        label=legend_label,
    )
    plt.axhline(
        unknowns,
        linestyle="--",
        label=f"unknowns = {unknowns}"
    )
    plt.xlabel("Number of Jacobian rows / gradients used")
    plt.ylabel("Average Jacobian rank")
    plt.title(f"Jacobian rank sweep: {args.network}, {args.dataset}, {args.method}")
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(save_dir, base + ".png")
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"\nSaved CSV to: {csv_path}")
    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()