import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from Misc_functions import get_keep_ids, compute_psnr_from_mse, build_network, get_keep_ids_by_gradsize
from Network import weights_init

def run_single_experiment(idx_net, device_id, dst, dataset_name, config, result_queue):
    """Runs a single experiment on an assigned GPU"""
    device = f'cuda:{device_id}'
    
    # Unpack config
    channel = config['channel']
    num_classes = config['num_classes']
    shape_img = config['shape_img']
    lr = config['lr']
    num_dummy = config['num_dummy']
    Iteration = config['Iteration']
    MASK_MODE = config['MASK_MODE']
    GRADSIZE_TOPK = config['GRADSIZE_TOPK']
    GRADSIZE_TOPFRAC = config['GRADSIZE_TOPFRAC']
    GRADSIZE_THRESHOLD = config['GRADSIZE_THRESHOLD']
    GRADSIZE_METRIC = config['GRADSIZE_METRIC']
    NETWORK_NAME = config['NETWORK_NAME']
    
    net = build_network(NETWORK_NAME, channel=channel, num_classes=num_classes, input_size=shape_img)
    net.apply(weights_init)
    net = net.to(device)

    print(f'[GPU {device_id}] Running {idx_net} experiment')
    
    idx_shuffle = np.random.permutation(len(dst))
    tt = transforms.Compose([transforms.ToTensor()])
    tp = transforms.Compose([transforms.ToPILImage()])

    final_recon = {}

    for method in ['iDLG', 'iDLG_masked']:
        print(f'[GPU {device_id}] {method}, Try to generate {num_dummy} images')

        criterion = nn.CrossEntropyLoss().to(device)
        imidx_list = []

        # ---- build GT batch ----
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

        # iDLG label inference
        label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape((1,))

        # choose which gradient tensors are "shared"
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

            losses.append(current_loss)
            mses.append(torch.mean((dummy_data - gt_data) ** 2).item())

            if iters % 100 == 0:
                print(f'[GPU {device_id}] iters {iters}, loss = {current_loss:.8f}, mse = {mses[-1]:.8f}')

            if current_loss < 1e-6:
                break

        final_recon[method] = dummy_data.detach().clone()

        if method == 'iDLG':
            loss_iDLG = losses
            label_iDLG = label_pred.item()
            mse_iDLG = mses
        else:
            loss_iDLG_masked = losses
            label_iDLG_masked = label_pred.item()
            mse_iDLG_masked = mses

    # Prepare results to send back
    result = {
        'idx_net': idx_net,
        'gt_data': gt_data.detach().cpu().numpy(),
        'final_recon': {k: v.detach().cpu().numpy() for k, v in final_recon.items()},
        'psnr_idlg': compute_psnr_from_mse(mse_iDLG[-1], max_val=1.0),
        'psnr_masked': compute_psnr_from_mse(mse_iDLG_masked[-1], max_val=1.0),
        'loss_iDLG': loss_iDLG[-1],
        'mse_iDLG': mse_iDLG[-1],
        'loss_iDLG_masked': loss_iDLG_masked[-1],
        'mse_iDLG_masked': mse_iDLG_masked[-1],
        'label_iDLG': label_iDLG,
        'label_iDLG_masked': label_iDLG_masked,
        'gt_label': gt_label.detach().cpu().numpy(),
        'imidx_list': imidx_list,
    }
    
    result_queue.put(result)