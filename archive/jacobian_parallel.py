import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import functions.consts as consts
from functions.Dataset import load_dataset
from functions.io_utils import parse_prefixes_with_fracs
from helper.Network import get_model, weights_init
from functions.masking import build_gradient_mask
from helper.metrics import compute_jacobian_rank


def split_list(lst, n_chunks):
    """Split lst into n_chunks as evenly as possible."""
    chunks = [[] for _ in range(n_chunks)]
    for i, x in enumerate(lst):
        chunks[i % n_chunks].append(x)
    return chunks


def worker(rank, world_size, args, sample_chunks, shared_results, progress_counter, progress_lock):
    """Per-worker function: computes Jacobian rank for each assigned sample."""
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())

    if args.device.startswith("cuda") and torch.cuda.is_available():
        device = f"cuda:{rank % torch.cuda.device_count()}"
    else:
        device = "cpu"

    seed = args.run_id + 1 + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    row_counts = [int(x.strip()) for x in args.row_counts.split(",") if x.strip()]
    prefixes, prefix_layer_fracs = parse_prefixes_with_fracs(args.prefixes)

    if os.access('/work3/s234843/bachelor', os.R_OK | os.W_OK | os.X_OK):
        data_path = '/work3/s234843/bachelor/datasets'
    else:
        data_path = './data'

    dst, channel, num_classes, shape_img = load_dataset(args.dataset, data_path)

    net = get_model(args.network, channel=channel, num_classes=num_classes,
                    input_size=shape_img, pretrained=args.pretrained)
    if not args.pretrained and args.network in ("LeNet", "LeNet_bigger", "MediumCNN", "BiggerCNN"):
        net.apply(weights_init)
    net = net.to(device).double()
    net.eval()

    tt = transforms.Compose([transforms.ToTensor()])
    criterion = nn.CrossEntropyLoss().to(device)

    if args.pretrained and channel == 3:
        dm = torch.tensor(consts.imagenet_mean, device=device, dtype=torch.float64).view(1, channel, 1, 1)
        ds = torch.tensor(consts.imagenet_std, device=device, dtype=torch.float64).view(1, channel, 1, 1)
    else:
        dm = torch.tensor(getattr(consts, f'{args.dataset.lower()}_mean'), device=device, dtype=torch.float64).view(1, channel, 1, 1)
        ds = torch.tensor(getattr(consts, f'{args.dataset.lower()}_std'), device=device, dtype=torch.float64).view(1, channel, 1, 1)

    local_results = {rows: [] for rows in row_counts}
    my_indices = sample_chunks[rank]

    for local_i, idx in enumerate(my_indices):
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
            gradsize_metric=args.gradsize_metric,
        )

        if rank == 0 and local_i == 0:
            named_params = list(net.named_parameters())
            print("\n=== MASK DEBUG ===", flush=True)
            print("mask_mode:", args.mask_mode, flush=True)
            print("prefixes:", prefixes, flush=True)
            print("prefix_layer_fracs:", prefix_layer_fracs, flush=True)
            print("entry_masks is None:", entry_masks is None, flush=True)
            print("num keep_ids:", 0 if keep_ids is None else len(keep_ids), flush=True)
            if keep_ids is not None:
                keep_ids_set = set(keep_ids)
                total_entries = sum(g.numel() for g in original_dy_dx if g is not None)
                observed_entries = sum(
                    g.numel() for i, g in enumerate(original_dy_dx)
                    if g is not None and i in keep_ids_set
                )
                print(f"observed_entries={observed_entries}, total_entries={total_entries}, "
                      f"kept_fraction={observed_entries/total_entries:.6f}", flush=True)
                for prefix in prefixes:
                    total = kept = 0
                    kept_names = []
                    for i, (name, _) in enumerate(named_params):
                        if name.startswith(prefix):
                            total += 1
                            if i in keep_ids_set:
                                kept += 1
                                kept_names.append(name)
                    print(f"  {prefix}: kept {kept}/{total}", flush=True)
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
            )
            local_results[rows].append(jac_rank)

            with progress_lock:
                progress_counter.value += 1

        del gt_data, gt_label, gt_data_norm, out, loss, dy_dx, original_dy_dx
        if torch.cuda.is_available() and device.startswith("cuda"):
            torch.cuda.empty_cache()

    shared_results[rank] = local_results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--network", type=str, default="resnet18")
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--method", type=str, default="idlg", choices=["idlg", "masked"])
    parser.add_argument("--pretrained", action="store_true",
                        help="Load ImageNet-pretrained weights (affects normalization).")

    parser.add_argument("--mask_mode", type=str, default="gradsize_topfrac")
    parser.add_argument("--prefixes", type=str, default="conv1:1.0,layer1:1.0,layer2:1.0,layer3:1.0,fc:1.0")
    parser.add_argument("--gradsize_topk", type=int, default=20)
    parser.add_argument("--gradsize_topfrac", type=float, default=0.5)
    parser.add_argument("--gradsize_metric", type=str, default="l2")

    parser.add_argument("--row_counts", type=str, default="3072,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000")
    parser.add_argument("--jacobian_select_mode", type=str, default="topk_abs",
                        choices=["topk_abs", "first", "random"])
    parser.add_argument("--rank_tol", type=float, default=1e-6,
                        help="Relative tolerance for numerical Jacobian rank.")

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
            "rows_used", "mean_rank", "std_rank",
            "unknowns", "num_samples", "jacobian_select_mode",
        ])
        for x, m, s in zip(xs, mean_ranks, std_ranks):
            writer.writerow([x, m, s, unknowns, args.num_samples, args.jacobian_select_mode])

    legend_label = f"mean rank (select={args.jacobian_select_mode}, samples={args.num_samples}"
    if args.method == "masked":
        legend_label += f", mask={args.mask_mode}"
    legend_label += ")"

    plt.figure(figsize=(7, 5))
    plt.errorbar(xs, mean_ranks, yerr=std_ranks, marker="o", capsize=4, label=legend_label)
    plt.axhline(unknowns, linestyle="--", label=f"unknowns = {unknowns}")
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
