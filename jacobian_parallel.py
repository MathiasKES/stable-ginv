import os
import csv
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

import consts
from Dataset import lfw_dataset
from Network import weights_init
from training_utils import build_network
from masking import build_gradient_mask
from metrics import compute_jacobian_rank


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


def split_list(lst, n_chunks):
    """Split lst into n_chunks as evenly as possible."""
    chunks = [[] for _ in range(n_chunks)]
    for i, x in enumerate(lst):
        chunks[i % n_chunks].append(x)
    return chunks


def worker(rank, world_size, args, sample_chunks, shared_results, progress_counter, progress_lock):
    """
    rank: worker id
    world_size: total number of workers
    sample_chunks: list of sample-index chunks, one per worker
    shared_results: manager dict for returning per-worker results
    """
    # Important for CUDA multiprocessing
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())

    # Decide device for this worker
    if args.device.startswith("cuda"):
        if torch.cuda.is_available():
            device = f"cuda:{rank % torch.cuda.device_count()}"
        else:
            device = "cpu"
    else:
        device = "cpu"

    # Per-worker seed
    seed = args.run_id + 1 + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    row_counts = [int(x.strip()) for x in args.row_counts.split(",") if x.strip()]

    prefixes_list = []
    prefix_layer_values = {}
    for item in args.prefixes.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            prefix, value = item.split(":", 1)
            prefix = prefix.strip()
            value = float(value.strip())
            prefixes_list.append(prefix)
            prefix_layer_values[prefix] = value
        else:
            prefixes_list.append(item)

    prefixes = tuple(prefixes_list)

    if os.access('/work3/s234843/bachelor', os.R_OK | os.W_OK | os.X_OK):
        data_path = '/work3/s234843/bachelor/datasets'
    else:
        data_path = './data'

    dst, channel, num_classes, shape_img = load_dataset(args.dataset, data_path)

    net = build_network(
        args.network,
        channel=channel,
        num_classes=num_classes,
        input_size=shape_img
    )
    if not args.network.startswith("resnet"):
        net.apply(weights_init)
    net = net.to(device).double()
    net.eval()

    tt = transforms.Compose([transforms.ToTensor()])
    criterion = nn.CrossEntropyLoss().to(device)

    if channel == 1:
        dm = torch.tensor(
            getattr(consts, f'{args.dataset.lower()}_mean'),
            device=device, dtype=torch.float64
        ).view(1, 1, 1, 1)
        ds = torch.tensor(
            getattr(consts, f'{args.dataset.lower()}_std'),
            device=device, dtype=torch.float64
        ).view(1, 1, 1, 1)
    else:
        dm = torch.tensor(
            getattr(consts, f'{args.dataset.lower()}_mean'),
            device=device, dtype=torch.float64
        ).view(1, channel, 1, 1)
        ds = torch.tensor(
            getattr(consts, f'{args.dataset.lower()}_std'),
            device=device, dtype=torch.float64
        ).view(1, channel, 1, 1)

    local_results = {rows: [] for rows in row_counts}
    my_indices = sample_chunks[rank]

    #print(f"[Worker {rank}] device={device}, samples={my_indices}", flush=True)

    for local_i, idx in enumerate(my_indices):
        #print(f"[Worker {rank}] sample {local_i+1}/{len(my_indices)}, dataset idx={idx}", flush=True)

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
            prefix_layer_fracs=prefix_layer_values,
            gradsize_topk=args.gradsize_topk,
            gradsize_topfrac=args.gradsize_topfrac,
            gradsize_threshold=args.gradsize_threshold,
            gradsize_metric=args.gradsize_metric,
        )

         # ----- debug masking -----
        if rank == 0 and local_i == 0:
            named_params = list(net.named_parameters())

            print("\n=== MASK DEBUG ===", flush=True)
            print("mask_mode:", args.mask_mode, flush=True)
            print("prefixes:", prefixes, flush=True)
            print("prefix_layer_values:", prefix_layer_values, flush=True)
            print("entry_masks is None:", entry_masks is None, flush=True)
            print("num keep_ids:", 0 if keep_ids is None else len(keep_ids), flush=True)

            if keep_ids is not None:
                keep_ids_set = set(keep_ids)

                total_entries = sum(g.numel() for g in original_dy_dx if g is not None)
                observed_entries = sum(
                    g.numel() for i, g in enumerate(original_dy_dx)
                    if g is not None and i in keep_ids_set
                )
                print(f"observed_entries={observed_entries}, total_entries={total_entries}, kept_fraction={observed_entries/total_entries:.6f}", flush=True)

                print("kept tensors per prefix:", flush=True)
                for prefix in prefixes:
                    total = 0
                    kept = 0
                    kept_names = []

                    for i, (name, _) in enumerate(named_params):
                        if name.startswith(prefix):
                            total += 1
                            if i in keep_ids_set:
                                kept += 1
                                kept_names.append(name)

                    requested = int(prefix_layer_values.get(prefix, args.gradsize_topk))
                    print(f"  {prefix}: kept {kept}/{total}, requested={requested}", flush=True)
                    for name in kept_names:
                        print(f"    - {name}", flush=True)

            print("=== END MASK DEBUG ===\n", flush=True)

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

            # print(
            #     f"[Worker {rank}] idx={idx}, rows={rows}, rank={jac_rank}, shape={jac_shape}",
            #     flush=True
            # )
            local_results[rows].append(jac_rank)

            with progress_lock:
                progress_counter.value += 1

        # Optional cleanup
        del gt_data, gt_label, gt_data_norm, out, loss, dy_dx, original_dy_dx
        if torch.cuda.is_available() and device.startswith("cuda"):
            torch.cuda.empty_cache()

    shared_results[rank] = local_results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--network", type=str, default="resnet18")
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--method", type=str, default="idlg", choices=["idlg", "masked"])

    parser.add_argument("--mask_mode", type=str, default="gradsize_topfrac")
    parser.add_argument("--prefixes", type=str, default="conv1,layer1,layer2,layer3,fc")
    parser.add_argument("--gradsize_topk", type=int, default=20)
    parser.add_argument("--gradsize_topfrac", type=float, default=0.5)
    parser.add_argument("--gradsize_threshold", type=float, default=None)
    parser.add_argument("--gradsize_metric", type=str, default="l2")

    parser.add_argument("--row_counts", type=str, default="3072,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000")
    parser.add_argument("--jacobian_select_mode", type=str, default="topk_abs",
                        choices=["topk_abs", "first", "random"])
    parser.add_argument("--rank_tol", type=float, default=0)
    parser.add_argument("--normalize_jacobian_rows", action="store_true")

    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=2)

    args = parser.parse_args()

    row_counts = [int(x.strip()) for x in args.row_counts.split(",") if x.strip()]

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
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    idx_shuffle = np.random.permutation(len(dst))
    sample_indices = idx_shuffle[:args.num_samples].tolist()

    unknowns = channel * shape_img[0] * shape_img[1]
    print(f"Unknowns: {unknowns}")
    print(f"Samples: {sample_indices}")

    world_size = min(args.num_workers, len(sample_indices))
    sample_chunks = split_list(sample_indices, world_size)

    manager = mp.Manager()
    shared_results = manager.dict()

    progress_counter = manager.Value("i", 0)
    progress_lock = manager.Lock()

    spawn_ctx = mp.spawn(
        worker,
        args=(world_size, args, sample_chunks, shared_results, progress_counter, progress_lock),
        nprocs=world_size,
        join=False
    )

    total_steps = args.num_samples * len(row_counts)
    last = 0

    with tqdm(total=total_steps, desc="Overall progress") as pbar:
        while not spawn_ctx.join(timeout=0.2):
            current = progress_counter.value
            if current > last:
                pbar.update(current - last)
                last = current

        current = progress_counter.value
        if current > last:
            pbar.update(current - last)

    rank_results = {rows: [] for rows in row_counts}
    for worker_id in range(world_size):
        worker_out = shared_results[worker_id]
        for rows in row_counts:
            rank_results[rows].extend(worker_out[rows])

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
    mp.set_start_method("spawn", force=True)
    main()