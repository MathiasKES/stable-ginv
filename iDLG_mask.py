import os
import numpy as np
import torch
from torchvision import datasets, transforms
from datetime import datetime
import csv
import os
import torch.multiprocessing as mp
import argparse

from Misc_functions import save_recon_panel
from Dataset import lfw_dataset
from run_single_exp import run_single_experiment


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mask_mode", type=str, default="gradsize_topfrac") # "gradsize_topk", "gradsize_topfrac", "gradsize_threshold", or "resnet_l1_fc", "conv12_fc"  # "all", "conv12", "conv123", "fc_only", "no_fc", "conv1_fc", "conv12_fc"
    parser.add_argument("--gradsize_topk", type=int, default=20)
    parser.add_argument("--gradsize_topfrac", type=float, default=0.5)
    parser.add_argument("--gradsize_threshold", type=float, default=None)
    parser.add_argument("--gradsize_metric", type=str, default="l2")

    parser.add_argument("--lr", type=float, default=1)
    parser.add_argument("--num_dummy", type=int, default=1)
    parser.add_argument("--iteration", type=int, default=1000)
    parser.add_argument("--num_exp", type=int, default=16)

    parser.add_argument("--network", type=str, default="MediumCNN")
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--run_id", type=int, default=0)

    args = parser.parse_args()

    # -------- Masking config --------
    MASK_MODE = args.mask_mode
    GRADSIZE_TOPK = args.gradsize_topk
    GRADSIZE_TOPFRAC = args.gradsize_topfrac
    GRADSIZE_THRESHOLD = args.gradsize_threshold
    GRADSIZE_METRIC = args.gradsize_metric
    lr = args.lr
    num_dummy = args.num_dummy
    Iteration = args.iteration
    num_exp = args.num_exp
    NETWORK_NAME = args.network
    dataset = args.dataset

    # -------- Masking config --------
    # MASK_MODE = "gradsize_topk"  # "gradsize_topk", "gradsize_topfrac", "gradsize_threshold", or "resnet_l1_fc", "conv12_fc"  # "all", "conv12", "conv123", "fc_only", "no_fc", "conv1_fc", "conv12_fc"
    # GRADSIZE_TOPK = 10
    # GRADSIZE_TOPFRAC = 0.6
    # GRADSIZE_THRESHOLD = None
    # GRADSIZE_METRIC = "l2"  # "l2", "mean_abs", "sum_abs"
    # lr = 1
    # num_dummy = 1
    # Iteration = 1000
    # num_exp = 16
    # NETWORK_NAME = "MediumCNN"  # options: "LeNet", "LeNet_bigger", "MediumCNN", "resnet20"
    # dataset = 'cifar100'
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    loss_tol = 1e-6
    patience = 50
    min_rel_improve = 1e-4
    explode_factor = 50.0
    warmup = 100
    max_nan = 1

    root_path = '.'
    if os.access('/work3/s234843/bachelor', os.R_OK | os.W_OK | os.X_OK):
        data_path = '/work3/s234843/bachelor/datasets'
        save_path = '/work3/s234843/bachelor/results'
    else:
        data_path = os.path.join(root_path, 'data').replace('\\', '/')
        save_path = os.path.join(root_path, 'results').replace('\\', '/')

    tt = transforms.Compose([transforms.ToTensor()])
    tp = transforms.Compose([transforms.ToPILImage()])

    try:
        os.makedirs(data_path, mode=0o770, exist_ok=True)
        os.makedirs(save_path, mode=0o770, exist_ok=True)
    except Exception as e:
        print(f"Warning: failed to set permissions for directories: {e}")

    print(dataset, 'root_path:', root_path)
    print(dataset, 'data_path:', data_path)
    print(dataset, 'save_path:', save_path)

    # -------- load data --------
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
        try:
            os.makedirs(lfw_path, mode=0o770, exist_ok=True)
        except Exception as e:
            print(f"Warning: failed to set permissions for {lfw_path}: {e}")
        dst = lfw_dataset(lfw_path, shape_img)
    else:
        raise ValueError('unknown dataset')


    # -------- panel buffers --------
    panel_block_size = num_exp
    panel_block_idx = 0
    panel_gt_pil = []
    panel_idlg_pil = []
    panel_masked_pil = []

    psnr_idlg_all = []
    psnr_masked_all = []
    final_loss_idlg_all = []
    final_mse_idlg_all = []
    final_loss_masked_all = []
    final_mse_masked_all = []

    mask_desc = MASK_MODE
    params = {"num-exp": num_exp, "lr": lr, "batchsize": num_dummy, "iters": Iteration}

    # Prepare config to pass to workers
    config = {
        'channel': channel,
        'num_classes': num_classes,
        'shape_img': shape_img,
        'lr': lr,
        'num_dummy': num_dummy,
        'Iteration': Iteration,
        'MASK_MODE': MASK_MODE,
        'GRADSIZE_TOPK': GRADSIZE_TOPK,
        'GRADSIZE_TOPFRAC': GRADSIZE_TOPFRAC,
        'GRADSIZE_THRESHOLD': GRADSIZE_THRESHOLD,
        'GRADSIZE_METRIC': GRADSIZE_METRIC,
        'NETWORK_NAME': NETWORK_NAME,
        'EarlyStop': {
            'loss_tol': loss_tol,
            'patience': patience,
            'min_rel_improve': min_rel_improve,
            'explode_factor': explode_factor,
            'warmup': warmup,
            'max_nan': max_nan,
        }
    }

    # -------- Run experiments in parallel --------
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs")
    if num_gpus == 0:
        raise RuntimeError("No CUDA GPUs available.")
    
    mp.set_start_method('spawn', force=True)
    mp.set_sharing_strategy('file_system')

    result_queue = mp.SimpleQueue()
    active_processes = {}
    next_exp = 0
    completed = 0

    # Start one experiment per GPU initially
    for device_id in range(min(num_gpus, num_exp)):
        p = mp.Process(
            target=run_single_experiment,
            args=(next_exp, device_id, dst, dataset, config, result_queue)
        )
        p.start()
        active_processes[device_id] = p
        print(f"Launching experiment {next_exp} on GPU {device_id}", flush=True)
        next_exp += 1

    # Keep launching a new experiment whenever one finishes
    while completed < num_exp:
        result = result_queue.get()
        completed += 1

        idx_net = result['idx_net']
        finished_device = result['device_id']

        psnr_idlg_all.append(result['psnr_idlg'])
        psnr_masked_all.append(result['psnr_masked'])
        final_loss_idlg_all.append(result['loss_iDLG'])
        final_mse_idlg_all.append(result['mse_iDLG'])
        final_loss_masked_all.append(result['loss_iDLG_masked'])
        final_mse_masked_all.append(result['mse_iDLG_masked'])

        # ---- accumulate recon panel ----
        gt_pil = tp(torch.from_numpy(result['gt_data'])[0])
        idlg_pil = tp(torch.from_numpy(result['final_recon']['iDLG'])[0])
        masked_pil = tp(torch.from_numpy(result['final_recon']['iDLG_masked'])[0])

        panel_gt_pil.append(gt_pil)
        panel_idlg_pil.append(idlg_pil)
        panel_masked_pil.append(masked_pil)

        if len(panel_gt_pil) == panel_block_size:
            save_recon_panel(
                params, panel_gt_pil, panel_idlg_pil, panel_masked_pil,
                save_path, panel_block_idx, dataset, mask_desc, timestamp_str
            )
            panel_block_idx += 1
            panel_gt_pil.clear()
            panel_idlg_pil.clear()
            panel_masked_pil.clear()

        es_r = result.get("early_stop_reason", {})
        es_i = result.get("early_stop_iter", {})
        print(f"early_stop iDLG: {es_r.get('iDLG')} @ {es_i.get('iDLG')}")
        print(f"early_stop masked: {es_r.get('iDLG_masked')} @ {es_i.get('iDLG_masked')}")
        print('imidx_list:', result['imidx_list'])
        print('loss_iDLG:', result['loss_iDLG'], 'loss_iDLG_masked:', result['loss_iDLG_masked'])
        print('mse_iDLG:', result['mse_iDLG'], 'mse_iDLG_masked:', result['mse_iDLG_masked'])
        print('gt_label:', result['gt_label'],
            'lab_iDLG:', result['label_iDLG'], 'lab_iDLG_masked:', result['label_iDLG_masked'])
        print('----------------------\n\n')

        # Clean up the finished process on that GPU
        active_processes[finished_device].join()

        # Start the next experiment immediately on the freed GPU
        if next_exp < num_exp:
            p = mp.Process(
                target=run_single_experiment,
                args=(next_exp, finished_device, dst, dataset, config, result_queue)
            )
            p.start()
            active_processes[finished_device] = p
            print(f"Launching experiment {next_exp} on GPU {finished_device}", flush=True)
            next_exp += 1

    # Final cleanup
    for p in active_processes.values():
        p.join()
    # save any remaining
    if len(panel_gt_pil) > 0:
        save_recon_panel(params, panel_gt_pil, panel_idlg_pil, panel_masked_pil,
                         save_path, panel_block_idx, dataset, mask_desc, timestamp_str)
    
    # -------- Compute statistics --------
    avg_psnr_idlg = float(np.mean(psnr_idlg_all)) if len(psnr_idlg_all) else float("nan")
    avg_psnr_masked = float(np.mean(psnr_masked_all)) if len(psnr_masked_all) else float("nan")
    
    avg_final_loss_idlg = float(np.mean(final_loss_idlg_all)) if final_loss_idlg_all else float("nan")
    avg_final_mse_idlg  = float(np.mean(final_mse_idlg_all)) if final_mse_idlg_all else float("nan")
    avg_final_loss_masked = float(np.mean(final_loss_masked_all)) if final_loss_masked_all else float("nan")
    avg_final_mse_masked  = float(np.mean(final_mse_masked_all)) if final_mse_masked_all else float("nan")

    med_final_loss_idlg = float(np.median(final_loss_idlg_all)) if final_loss_idlg_all else float("nan")
    med_final_mse_idlg  = float(np.median(final_mse_idlg_all)) if final_mse_idlg_all else float("nan")
    med_final_loss_masked = float(np.median(final_loss_masked_all)) if final_loss_masked_all else float("nan")
    med_final_mse_masked  = float(np.median(final_mse_masked_all)) if final_mse_masked_all else float("nan")

    csv_path = os.path.join(save_path, "exp_results.csv")
    file_exists = os.path.isfile(csv_path)

    common = {
        "timestamp": timestamp_str,
        "dataset": dataset,
        "network": NETWORK_NAME,
        
        "lr": lr,
        "iteration": Iteration,
        "num_exp": num_exp,}
    
    grad_value = ""
    if MASK_MODE == "gradsize_topfrac":
        grad_value = GRADSIZE_TOPFRAC
    elif MASK_MODE == "gradsize_topk":
        grad_value = GRADSIZE_TOPK

    rows = [
        {   "method": "iDLG",
            **common,
            "mask_mode": "",
            "grad_param": "",
            "med_final_loss": med_final_loss_idlg,
            "avg_final_loss": avg_final_loss_idlg,
            "med_final_mse": med_final_mse_idlg,
            "avg_final_mse": avg_final_mse_idlg,
            "avg_psnr": avg_psnr_idlg},

        {   "method": "iDLG_masked",
            **common,
            "mask_mode": MASK_MODE,
            "grad_param": grad_value,
            "med_final_loss": med_final_loss_masked,
            "avg_final_loss": avg_final_loss_masked,
            "med_final_mse": med_final_mse_masked,
            "avg_final_mse": avg_final_mse_masked,
            "avg_psnr": avg_psnr_masked}
    ]

    fieldnames = ["method"] + [k for k in common.keys()] + ["mask_mode", "grad_param", "med_final_loss", "avg_final_loss", "med_final_mse", "avg_final_mse", "avg_psnr"]

    # rows = [
    #     {"method": "iDLG", **common, "med_final_loss": med_final_loss_idlg, "avg_final_loss": avg_final_loss_idlg, "med_final_mse": med_final_mse_idlg, "avg_final_mse": avg_final_mse_idlg, "avg_psnr": avg_psnr_idlg},
    #     {"method": "iDLG_masked", **common, "mask_mode": MASK_MODE, "grad_topfrac": GRADSIZE_TOPFRAC, "med_final_loss": med_final_loss_masked, "avg_final_loss": avg_final_loss_masked, "med_final_mse": med_final_mse_masked, "avg_final_mse": avg_final_mse_masked, "avg_psnr": avg_psnr_masked},]
    
    # fieldnames = ["method"] + [k for k in common.keys()] + ["mask_mode", "grad_topfrac", "med_final_loss", "avg_final_loss", "med_final_mse", "avg_final_mse", "avg_psnr"]

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)
    
    try:
        os.chmod(csv_path, 0o770) # Ensure correct permissions
    except Exception as e:
        print(f"Warning: failed to set permissions for {csv_path}: {e}")

    print("\n=== Average PSNR over all experiments ===")
    print(f"\nSaved CSV rows to: {csv_path}")
    if MASK_MODE == "gradsize_topfrac":
        print(f"top fraction of gradient sizes kept: {GRADSIZE_TOPFRAC*100}%")
    elif MASK_MODE == "gradsize_topk":
        print(f"top-k gradient sizes kept: {GRADSIZE_TOPK}")
    print(f"Avg final loss iDLG: {avg_final_loss_idlg:.6f} | masked: {avg_final_loss_masked:.6f}")
    print(f"Avg final mse  iDLG: {avg_final_mse_idlg:.8f} | masked: {avg_final_mse_masked:.8f}")
    print(f"Median final loss iDLG: {med_final_loss_idlg:.6f} | masked: {med_final_loss_masked:.6f}")
    print(f"Median final mse  iDLG: {med_final_mse_idlg:.8f} | masked: {med_final_mse_masked:.8f}")
    print(f"Average PSNR iDLG: {avg_psnr_idlg:.4f} dB | masked: {avg_psnr_masked:.4f} dB")

if __name__ == '__main__':
    main()