import time
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import pickle
import PIL.Image as Image
import math
from datetime import datetime

from Figures import save_recon_panel, compute_psnr_from_mse
from Network import LeNet, LeNetCIFAR10, LeNetCIFAR100, weights_init
from Dataset import Dataset_from_Image, lfw_dataset

def main():
    dataset = 'cifar10'
    root_path = ''
    data_path = os.path.join(root_path, 'data').replace('\\', '/')
    save_path = os.path.join(root_path, 'results/iDLG_%s'%dataset).replace('\\', '/')
    
    lr = 0.5
    num_dummy = 1
    Iteration = 300
    num_exp = 10

    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'

    tt = transforms.Compose([transforms.ToTensor()])
    tp = transforms.Compose([transforms.ToPILImage()])

    print(dataset, 'root_path:', root_path)
    print(dataset, 'data_path:', data_path)
    print(dataset, 'save_path:', save_path)

    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    ''' load data '''
    if dataset == 'MNIST':
        shape_img = (28, 28)
        num_classes = 10
        channel = 1
        hidden = 588
        dst = datasets.MNIST(data_path, download=True)

    elif dataset == 'cifar100':
        shape_img = (32, 32)
        num_classes = 100
        channel = 3
        hidden = 768
        dst = datasets.CIFAR100(data_path, download=True)
    
    elif dataset == 'cifar10':
        shape_img = (32, 32)
        num_classes = 10
        channel = 3
        hidden = 768
        dst = datasets.CIFAR10(data_path, download=True)


    elif dataset == 'lfw':
        shape_img = (32, 32)
        num_classes = 5749
        channel = 3
        hidden = 768
        lfw_path = os.path.join(root_path, '../data/lfw')
        dst = lfw_dataset(lfw_path, shape_img)

    else:
        exit('unknown dataset')

    panel_block_size = 10
    panel_block_idx = 0
    panel_gt_pil = []
    panel_idlg_pil = []
    panel_masked_pil = []
    mask_desc = "keep_params_0-3"  # update this when you change masking
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    ''' train iDLG and iDLG_masked '''

    # Collect PSNR curves for all experiments so we can plot them in a single grid at the end.
    all_psnr_curves = []  # [{'iDLG': {'iters': [...], 'psnr': [...]}, 'iDLG_masked': {...}}, ...]

    params = {"num-exp": num_exp, "lr": lr, "batchsize": num_dummy, "iters": Iteration}

    for idx_net in range(num_exp):
        net = LeNetCIFAR10(channel=channel, num_classes=num_classes)
        net.apply(weights_init)

        print('running %d|%d experiment'%(idx_net, num_exp))
        net = net.to(device)
        idx_shuffle = np.random.permutation(len(dst))

        curves = {
            'iDLG': {'loss': None, 'mse': None, 'psnr': None, 'iters': None},
            'iDLG_masked': {'loss': None, 'mse': None, 'psnr': None, 'iters': None},}
        
        final_recon = {}
        
        for method in ['iDLG', 'iDLG_masked']:
            print('%s, Try to generate %d images' % (method, num_dummy))

            criterion = nn.CrossEntropyLoss().to(device)
            imidx_list = []

            for imidx in range(num_dummy):
                idx = idx_shuffle[imidx]
                imidx_list.append(idx)
                tmp_datum = tt(dst[idx][0]).float().to(device)
                tmp_datum = tmp_datum.view(1, *tmp_datum.size())
                tmp_label = torch.Tensor([dst[idx][1]]).long().to(device)
                tmp_label = tmp_label.view(1, )
                if imidx == 0:
                    gt_data = tmp_datum
                    gt_label = tmp_label
                else:
                    gt_data = torch.cat((gt_data, tmp_datum), dim=0)
                    gt_label = torch.cat((gt_label, tmp_label), dim=0)


            # compute original gradient
            out = net(gt_data)
            y = criterion(out, gt_label)
            dy_dx = torch.autograd.grad(y, net.parameters())
            original_dy_dx = list((_.detach().clone() for _ in dy_dx))

            # generate dummy data and label
            dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
            optimizer = torch.optim.LBFGS([dummy_data, ], lr=lr,)

            # predict the ground-truth label
            label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape((1,)).requires_grad_(False)

            history = []
            history_iters = []
            losses = []
            mses = []
            train_iters = []
            psnrs = []

            #print('lr =', lr)
            for iters in range(Iteration):

                def closure():
                    optimizer.zero_grad()
                    pred = net(dummy_data)

                    dummy_loss = criterion(pred, label_pred)
                    dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
                    grad_diff = 0
                    if method == 'iDLG':
                        for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                            grad_diff += ((gx - gy) ** 2).sum()

                    elif method == 'iDLG_masked':
                        # -------- MASK: only first and last layer --------
                        L = len(original_dy_dx)
                        # First layer params: index 0 (weight) and 1 (bias)
                        # Last layer params: index L-2 (weight) and L-1 (bias)
                        keep_param_indices = {0,1,2,3}
                        for p_idx, (gx, gy) in enumerate(zip(dummy_dy_dx, original_dy_dx)):
                            if p_idx not in keep_param_indices:
                                continue
                            grad_diff += ((gx - gy) ** 2).sum()

                    grad_diff.backward()
                    return grad_diff
                
                loss_tensor = optimizer.step(closure)
                current_loss = float(loss_tensor)

                with torch.no_grad():
                    dummy_data.clamp_(0, 1)

                train_iters.append(iters)
                losses.append(current_loss)
                mse_val = torch.mean((dummy_data - gt_data) ** 2).item()
                mses.append(mse_val)
                psnrs.append(compute_psnr_from_mse(mse_val, max_val=1.0))

                if iters % 100 == 0:
                    current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
                    print(current_time, iters, 'loss = %.8f, mse = %.8f, psnr = %.4f dB' % (current_loss, mses[-1], psnrs[-1]))
                    history.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])
                    history_iters.append(iters)

                    for imidx in range(num_dummy):
                        plt.figure(figsize=(12, 8))
                        plt.subplot(3, 10, 1)
                        plt.imshow(tp(gt_data[imidx].cpu()))
                        for i in range(min(len(history), 29)):
                            plt.subplot(3, 10, i + 2)
                            plt.imshow(history[i][imidx])
                            plt.title('iter=%d' % (history_iters[i]))
                            plt.axis('off')
                        if method == 'iDLG':
                            #plt.savefig('%s/iDLG_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx]))
                            plt.close()
                        elif method == 'iDLG_masked':
                            #plt.savefig('%s/iDLG_masked_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx]))
                            plt.close()

                    if current_loss < 0.000001: # converge
                        break

            curves[method]['loss'] = losses
            curves[method]['mse'] = mses
            curves[method]['psnr'] = psnrs
            curves[method]['iters'] = train_iters
            final_recon[method] = dummy_data.detach().clone()

            if method == 'iDLG':
                loss_iDLG = losses
                label_iDLG = label_pred.item()
                mse_iDLG = mses
                psnr_iDLG = psnrs
            elif method == 'iDLG_masked':
                loss_iDLG_masked = losses
                label_iDLG_masked = label_pred.item()
                mse_iDLG_masked = mses
                psnr_iDLG_masked = psnrs

        # --- Accumulate a 10-experiment panel (GT / iDLG / iDLG_masked) ---
        # Using first (and only) dummy image: index 0
        gt_pil = tp(gt_data[0].detach().cpu())
        idlg_pil = tp(final_recon['iDLG'][0].detach().cpu())
        masked_pil = tp(final_recon['iDLG_masked'][0].detach().cpu())

        panel_gt_pil.append(gt_pil)
        panel_idlg_pil.append(idlg_pil)
        panel_masked_pil.append(masked_pil)

        # Save after every 10 experiments
        if len(panel_gt_pil) == panel_block_size:
            save_recon_panel(params, panel_gt_pil, panel_idlg_pil, panel_masked_pil, save_path, panel_block_idx, dataset, mask_desc, timestamp_str)
            panel_block_idx += 1
            panel_gt_pil.clear()
            panel_idlg_pil.clear()
            panel_masked_pil.clear()

        print('imidx_list:', imidx_list)
        print('loss_iDLG:', loss_iDLG[-1], 'loss_iDLG_masked:', loss_iDLG_masked[-1])
        print('mse_iDLG:', mse_iDLG[-1], 'mse_iDLG_masked:', mse_iDLG_masked[-1])
        print('psnr_iDLG:', psnr_iDLG[-1], 'psnr_iDLG_masked:', psnr_iDLG_masked[-1])
        print('gt_label:', gt_label.detach().cpu().data.numpy(), 'lab_iDLG:', label_iDLG, 'lab_iDLG_masked:', label_iDLG_masked)

        # Store curves for the final grid plot.
        all_psnr_curves.append({
            'iDLG': {'iters': curves['iDLG']['iters'], 'psnr': curves['iDLG']['psnr']},
            'iDLG_masked': {'iters': curves['iDLG_masked']['iters'], 'psnr': curves['iDLG_masked']['psnr']},
        })

        print('----------------------\n\n')

    # Save any remaining experiments (<10) once at the end
    if len(panel_gt_pil) > 0:
        save_recon_panel(params, panel_gt_pil, panel_idlg_pil, panel_masked_pil,
                        save_path, panel_block_idx, dataset, mask_desc, timestamp_str)

    # ---- Plot ALL PSNR curves in a single grid figure ----
    if len(all_psnr_curves) > 0:
        ncols = int(math.ceil(math.sqrt(num_exp)))
        nrows = int(math.ceil(num_exp / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)
        axes = axes.flatten()

        for i, curves_i in enumerate(all_psnr_curves):
            ax = axes[i]
            ax.plot(curves_i['iDLG']['iters'], curves_i['iDLG']['psnr'], label='iDLG')
            ax.plot(curves_i['iDLG_masked']['iters'], curves_i['iDLG_masked']['psnr'], label='iDLG_masked')
            ax.set_title(f'exp {i + 1}/{num_exp}')
            ax.set_xlabel('Iter')
            ax.set_ylabel('PSNR (dB)')
            ax.grid(True, alpha=0.3)
            # Keep legends readable by not repeating them on every subplot.
            if i == 0:
                ax.legend(fontsize=9)

        # Hide unused subplots (if num_exp is not a perfect grid fill).
        for j in range(len(all_psnr_curves), len(axes)):
            axes[j].axis('off')

        fig.suptitle(f'PSNR vs Iteration (dataset={dataset})', y=1.02)
        fig.tight_layout()
        psnr_grid_path = os.path.join(save_path, 'psnr_curves_grid.png')
        fig.savefig(psnr_grid_path, dpi=220, bbox_inches='tight')
        plt.close(fig)
        print('Saved PSNR grid plot to:', psnr_grid_path)

if __name__ == '__main__':
    main()


