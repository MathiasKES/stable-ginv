import os
import numpy as np
import torch
from torchvision import datasets, transforms
from datetime import datetime
import csv
import os
import torch.multiprocessing as mp

from Misc_functions import save_recon_panel
from Dataset import lfw_dataset
from run_single_exp import run_single_experiment


def main():
    # -------- Masking config --------
    MASK_MODE = "gradsize_topfrac"  # "gradsize_topk", "gradsize_topfrac", "gradsize_threshold", or "conv12_fc"  # "all", "conv12", "conv123", "fc_only", "no_fc", "conv1_fc", "conv12_fc"
    GRADSIZE_TOPK = 10
    GRADSIZE_TOPFRAC = 0.9
    GRADSIZE_THRESHOLD = None
    GRADSIZE_METRIC = "l2"  # "l2", "mean_abs", "sum_abs"
    lr = 1
    num_dummy = 1
    Iteration = 300
    num_exp = 8
    NETWORK_NAME = "LeNet_bigger"  # options: "LeNet", "LeNet_bigger", "MediumCNN"

    

    dataset = 'cifar100'
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    root_path = '.'
    if os.access('/work3/s234843/bachelor', os.R_OK | os.W_OK | os.X_OK):
        data_path = '/work3/s234843/bachelor/datasets'
        save_path = '/work3/s234843/bachelor/results'
    else:
        data_path = os.path.join(root_path, 'data').replace('\\', '/')
        save_path = os.path.join(root_path, 'results').replace('\\', '/')

    tt = transforms.Compose([transforms.ToTensor()])
    tp = transforms.Compose([transforms.ToPILImage()])

    os.makedirs(data_path, mode=0o770, exist_ok=True)
    os.makedirs(save_path, mode=0o770, exist_ok=True)

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
        lfw_path = os.path.join(root_path, '../data/lfw')
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
    }

    # -------- Run experiments in parallel --------
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs")

    mp.set_start_method('spawn', force=True)
    mp.set_sharing_strategy('file_system')

    result_queue = mp.SimpleQueue()
    processes = []

    # Spawn all experiments
    for idx_net in range(num_exp):
        device_id = idx_net % num_gpus
        p = mp.Process(target=run_single_experiment, 
                       args=(idx_net, device_id, dst, dataset, config, result_queue))
        p.start()
        processes.append(p)

    # Collect results as they complete
    for _ in range(num_exp):
        result = result_queue.get()
        
        idx_net = result['idx_net']
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

        # save after every block_size experiments
        if len(panel_gt_pil) == panel_block_size:
            save_recon_panel(params, panel_gt_pil, panel_idlg_pil, panel_masked_pil,
                             save_path, panel_block_idx, dataset, mask_desc, timestamp_str)
            panel_block_idx += 1
            panel_gt_pil.clear()
            panel_idlg_pil.clear()
            panel_masked_pil.clear()

        print('imidx_list:', result['imidx_list'])
        print('loss_iDLG:', result['loss_iDLG'], 'loss_iDLG_masked:', result['loss_iDLG_masked'])
        print('mse_iDLG:', result['mse_iDLG'], 'mse_iDLG_masked:', result['mse_iDLG_masked'])
        print('gt_label:', result['gt_label'],
              'lab_iDLG:', result['label_iDLG'], 'lab_iDLG_masked:', result['label_iDLG_masked'])
        print('----------------------\n\n')

    # Wait for all processes to finish
    for p in processes:
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

    rows = [
        {"method": "iDLG", **common, "med_final_loss": med_final_loss_idlg, "avg_final_loss": avg_final_loss_idlg, "med_final_mse": med_final_mse_idlg, "avg_final_mse": avg_final_mse_idlg, "avg_psnr": avg_psnr_idlg},
        {"method": "iDLG_masked", **common, "mask_mode": MASK_MODE, "grad_topfrac": GRADSIZE_TOPFRAC, "med_final_loss": med_final_loss_masked, "avg_final_loss": avg_final_loss_masked, "med_final_mse": med_final_mse_masked, "avg_final_mse": avg_final_mse_masked, "avg_psnr": avg_psnr_masked},]
    
    fieldnames = ["method"] + [k for k in common.keys()] + ["mask_mode", "grad_topfrac", "med_final_loss", "avg_final_loss", "med_final_mse", "avg_final_mse", "avg_psnr"]

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    print("\n=== Average PSNR over all experiments ===")
    print(f"\nSaved CSV rows to: {csv_path}")
    print(f"top fraction of gradients sizes kept: {GRADSIZE_TOPFRAC*100}%")
    print(f"Avg final loss iDLG: {avg_final_loss_idlg:.6f} | masked: {avg_final_loss_masked:.6f}")
    print(f"Avg final mse  iDLG: {avg_final_mse_idlg:.8f} | masked: {avg_final_mse_masked:.8f}")
    print(f"Median final loss iDLG: {med_final_loss_idlg:.6f} | masked: {med_final_loss_masked:.6f}")
    print(f"Median final mse  iDLG: {med_final_mse_idlg:.8f} | masked: {med_final_mse_masked:.8f}")
    print(f"Average PSNR iDLG: {avg_psnr_idlg:.4f} dB | masked: {avg_psnr_masked:.4f} dB")

if __name__ == '__main__':
    main()