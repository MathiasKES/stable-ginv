import time
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import PIL.Image as Image
from datetime import datetime
import csv

from Misc_functions import save_recon_panel, get_keep_ids, compute_psnr_from_mse, build_network, get_keep_ids_by_gradsize
from Dataset import Dataset_from_Image, lfw_dataset
from Network import LeNet, LeNet_bigger, MediumCNN, weights_init

# -------- Masking config --------
MASK_MODE = "gradsize_topfrac"  # "gradsize_topk", "gradsize_topfrac", "gradsize_threshold", or "conv12_fc"  # "all", "conv12", "conv123", "fc_only", "no_fc", "conv1_fc", "conv12_fc"
GRADSIZE_TOPK = 10
GRADSIZE_TOPFRAC = 0.9
GRADSIZE_THRESHOLD = None
GRADSIZE_METRIC = "l2"  # "l2", "mean_abs", "sum_abs"
lr = 1
num_dummy = 1
Iteration = 300
num_exp = 50
NETWORK_NAME = "LeNet"  # options: "LeNet", "LeNet_bigger", "MediumCNN"
# --------------------------------------------------

def main():
    dataset = 'MNIST'
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    root_path = '.'
    data_path = os.path.join(root_path, '../data').replace('\\', '/')
    save_path = os.path.join(root_path, f'results/iDLG_{dataset}').replace('\\', '/')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: ",device)
    tt = transforms.Compose([transforms.ToTensor()])
    tp = transforms.Compose([transforms.ToPILImage()])

    print(dataset, 'root_path:', root_path)
    print(dataset, 'data_path:', data_path)
    print(dataset, 'save_path:', save_path)

    os.makedirs('results', exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

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

    mask_desc = MASK_MODE  # auto label the saved panel
    params = {"num-exp": num_exp, "lr": lr, "batchsize": num_dummy, "iters": Iteration}

    # -------- train iDLG and iDLG_masked --------
    for idx_net in range(num_exp):
        net =  build_network(NETWORK_NAME, channel=channel, num_classes=num_classes, input_size=shape_img)
        net.apply(weights_init)
        net = net.to(device)

        print('running %d|%d experiment' % (idx_net, num_exp))
        idx_shuffle = np.random.permutation(len(dst))

        final_recon = {}

        for method in ['iDLG', 'iDLG_masked']:
            print('%s, Try to generate %d images' % (method, num_dummy))

            criterion = nn.CrossEntropyLoss().to(device)
            imidx_list = []

            # ---- build GT batch (size = num_dummy) ----
            for imidx in range(num_dummy):
                idx = idx_shuffle[imidx]
                imidx_list.append(idx)

                tmp_datum = tt(dst[idx][0]).float().to(device)
                tmp_datum = tmp_datum.view(1, *tmp_datum.size())
                tmp_label = torch.tensor([dst[idx][1]], dtype=torch.long, device=device).view(1,)

                if imidx == 0:
                    gt_data = tmp_datum
                    gt_label = tmp_label
                else:
                    gt_data = torch.cat((gt_data, tmp_datum), dim=0)
                    gt_label = torch.cat((gt_label, tmp_label), dim=0)

            # ---- compute original gradients ----
            out = net(gt_data)
            y = criterion(out, gt_label)
            dy_dx = torch.autograd.grad(y, net.parameters())
            original_dy_dx = [g.detach().clone() for g in dy_dx]

            # ---- dummy init ----
            dummy_data = torch.randn(gt_data.size(), device=device, requires_grad=True)
            optimizer = torch.optim.LBFGS([dummy_data], lr=lr)

            # iDLG label inference (uses fc.weight grad)
            label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape((1,))

            # choose which gradient tensors are "shared"
            #keep_ids = get_keep_ids("all" if method == "iDLG" else MASK_MODE)
            if method == "iDLG":
                keep_ids = get_keep_ids("all")
            else:
                if MASK_MODE == "gradsize_topk":
                    keep_ids, ranked = get_keep_ids_by_gradsize(
                        original_dy_dx, mode="topk", topk=GRADSIZE_TOPK, metric=GRADSIZE_METRIC)
                elif MASK_MODE == "gradsize_topfrac":
                    keep_ids, ranked = get_keep_ids_by_gradsize(
                        original_dy_dx, mode="topfrac", top_frac=GRADSIZE_TOPFRAC, metric=GRADSIZE_METRIC)
                elif MASK_MODE == "gradsize_threshold":
                    keep_ids, ranked = get_keep_ids_by_gradsize(
                        original_dy_dx, mode="threshold", threshold=GRADSIZE_THRESHOLD, metric=GRADSIZE_METRIC)
                else:
                    # fallback to your existing hand-crafted layer masks
                    keep_ids = get_keep_ids(MASK_MODE)
                    ranked = None
            losses = []
            mses = []

            for iters in range(Iteration):

                def closure():
                    optimizer.zero_grad()
                    pred = net(dummy_data)
                    dummy_loss = criterion(pred, label_pred)
                    dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
                    
                    grad_diff = 0.0
                    for i, (gx, gy) in enumerate(zip(dummy_dy_dx, original_dy_dx)):
                        if i not in keep_ids:
                            continue
                        grad_diff = grad_diff + ((gx - gy) ** 2).sum()
                    grad_diff.backward()
                    return grad_diff
                
                optimizer.step(closure)
                current_loss = optimizer.step(closure).item()

                #with torch.no_grad():
                #    dummy_data.clamp_(0, 1)

                losses.append(current_loss)
                mses.append(torch.mean((dummy_data - gt_data) ** 2).item())

                if iters % 50 == 0:
                    current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
                    print(current_time, iters, f'loss = {current_loss:.8f}, mse = {mses[-1]:.8f}')

                if current_loss < 1e-6:
                    break

            # final iterate (last LBFGS update) for this method
            final_recon[method] = dummy_data.detach().clone()

            if method == 'iDLG':
                loss_iDLG = losses
                label_iDLG = label_pred.item()
                mse_iDLG = mses
            else:
                loss_iDLG_masked = losses
                label_iDLG_masked = label_pred.item()
                mse_iDLG_masked = mses

        # --- PSNR for this experiment (use final MSE) ---
        psnr_idlg_all.append(compute_psnr_from_mse(mse_iDLG[-1], max_val=1.0))
        psnr_masked_all.append(compute_psnr_from_mse(mse_iDLG_masked[-1], max_val=1.0))
        
        final_loss_idlg_all.append(float(loss_iDLG[-1]))
        final_mse_idlg_all.append(float(mse_iDLG[-1]))
        final_loss_masked_all.append(float(loss_iDLG_masked[-1]))
        final_mse_masked_all.append(float(mse_iDLG_masked[-1]))


        # ---- accumulate recon panel (final iterates) ----
        gt_pil = tp(gt_data[0].detach().cpu())
        idlg_pil = tp(final_recon['iDLG'][0].detach().cpu())
        masked_pil = tp(final_recon['iDLG_masked'][0].detach().cpu())

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

        print('imidx_list:', imidx_list)
        print('loss_iDLG:', loss_iDLG[-1], 'loss_iDLG_masked:', loss_iDLG_masked[-1])
        print('mse_iDLG:', mse_iDLG[-1], 'mse_iDLG_masked:', mse_iDLG_masked[-1])
        print('gt_label:', gt_label.detach().cpu().numpy(),
              'lab_iDLG:', label_iDLG, 'lab_iDLG_masked:', label_iDLG_masked)
        print('----------------------\n\n')

    # save any remaining (< block_size) at end
    if len(panel_gt_pil) > 0:
        save_recon_panel(params, panel_gt_pil, panel_idlg_pil, panel_masked_pil,
                         save_path, panel_block_idx, dataset, mask_desc, timestamp_str)
    
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
        "mask_mode": MASK_MODE,
        "grad_topfrac": GRADSIZE_TOPFRAC,
        "lr": lr,
        "iteration": Iteration,
        "num_exp": num_exp,
        "num_dummy": num_dummy,}

    rows = [
        {"method": "iDLG", **common, "med_final_loss": med_final_loss_idlg, "avg_final_loss": avg_final_loss_idlg, "med_final_mse": med_final_mse_idlg, "avg_final_mse": avg_final_mse_idlg, "avg_psnr": avg_psnr_idlg},
        {"method": "iDLG_masked", **common, "med_final_loss": med_final_loss_masked, "avg_final_loss": avg_final_loss_masked, "med_final_mse": med_final_mse_masked, "avg_final_mse": avg_final_mse_masked, "avg_psnr": avg_psnr_masked},]
    
    fieldnames = ["method"] + [k for k in common.keys()] + ["med_final_loss", "avg_final_loss", "med_final_mse", "avg_final_mse", "avg_psnr"]

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    print("\n=== Average PSNR over all experiments ===")
    print(f"\nSaved CSV rows to: {csv_path}")
    print(f"top fraction of gradients sizes kept: {GRADSIZE_TOPFRAC}%")
    print(f"Avg final loss iDLG: {avg_final_loss_idlg:.6f} | masked: {avg_final_loss_masked:.6f}")
    print(f"Avg final mse  iDLG: {avg_final_mse_idlg:.8f} | masked: {avg_final_mse_masked:.8f}")
    print(f"Median final loss iDLG: {med_final_loss_idlg:.6f} | masked: {med_final_loss_masked:.6f}")
    print(f"Median final mse  iDLG: {med_final_mse_idlg:.8f} | masked: {med_final_mse_masked:.8f}")
    print(f"Average PSNR iDLG: {avg_psnr_idlg:.4f} dB | masked: {avg_psnr_masked:.4f} dB")

if __name__ == '__main__':
    main()