import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision
import os 
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "invertinggradients"))
import consts

from Misc_functions import get_keep_ids, compute_psnr_from_mse, build_network, get_keep_ids_by_gradsize
from Network import weights_init

def run_single_experiment(idx_net, device_id, dst, dataset_name, config, result_queue):
    """Runs a single experiment on an assigned GPU"""
    torch.cuda.set_device(device_id)
    device = f'cuda:{device_id}'
    
    # Unpack config
    channel: int = config['channel']
    num_classes: int = config['num_classes']
    shape_img: tuple[int] = config['shape_img']
    lr: float = config['lr']
    num_dummy: int = config['num_dummy']
    Iteration: int = config['Iteration']
    MASK_MODE: str = config['MASK_MODE']
    PREFIXES: str = config.get('PREFIXES', ())
    GRADSIZE_TOPK: int = config['GRADSIZE_TOPK']
    GRADSIZE_TOPFRAC: float = config['GRADSIZE_TOPFRAC']
    GRADSIZE_THRESHOLD: float = config['GRADSIZE_THRESHOLD']
    GRADSIZE_METRIC: str = config['GRADSIZE_METRIC']
    NETWORK_NAME: str = config['NETWORK_NAME']

    seed = config.get("run_id", 0) + idx_net
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    net = build_network(NETWORK_NAME, channel=channel, num_classes=num_classes, input_size=shape_img)
    if not NETWORK_NAME.startswith("resnet"):
        net.apply(weights_init)
    net = net.to(device)
    net.eval()

    print(f'[GPU {device_id}] Running {idx_net} experiment')
    
    idx_shuffle = np.random.permutation(len(dst))
    tt = transforms.Compose([transforms.ToTensor()])
    tp = transforms.Compose([transforms.ToPILImage()])

    final_recon = {}
    early_stop_reason_dict = {}
    early_stop_iter_dict = {}

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

        dm = torch.tensor(getattr(consts, f'{dataset_name.lower()}_mean'), device=device).view(1,3,1,1)
        ds = torch.tensor(getattr(consts, f'{dataset_name.lower()}_std'), device=device).view(1,3,1,1)

        # ---- compute original gradients ----
        gt_data_norm = (gt_data - dm) / ds # normalize gt data
        out = net(gt_data_norm)
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
            elif MASK_MODE == "prefix":
                keep_ids = get_keep_ids(mask_mode="prefix", net=net, prefixes=PREFIXES)
                ranked = None
            elif MASK_MODE == "resnet_l1_fc":
                keep_ids = get_keep_ids(mask_mode="prefix", net=net, prefixes=("layer1", "linear"))
            else:
                keep_ids = get_keep_ids(MASK_MODE)
                ranked = None
        
        losses = []
        mses = []
        # ---- EarlyStop config from main (with fallbacks) ----
        es = config.get("EarlyStop", {})
        best_loss = float("inf")
        no_improve = 0

        patience = int(es.get("patience", 50))
        min_rel_improve = float(es.get("min_rel_improve", 1e-4))
        explode_factor = float(es.get("explode_factor", 500.0))
        warmup = int(es.get("warmup", 100))
        loss_tol = float(es.get("loss_tol", 1e-6))
        max_nan = int(es.get("max_nan", 1))

        nan_count = 0
        early_stop_reason = None
        early_stop_iter = None
        best_loss_value = float("inf")
        best_dummy = None
        best_mse_value = None

        for iters in range(Iteration):
            def closure():
                optimizer.zero_grad()

                x = torch.sigmoid(dummy_data)
                x_norm = (x - dm) / ds # normalize before forward pass

                pred = net(x_norm)

                dummy_loss = criterion(pred, gt_label)
                dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
                
                grad_diff = 0.0
                for i, (gx, gy) in enumerate(zip(dummy_dy_dx, original_dy_dx)):
                    if i not in keep_ids:
                        continue
                    grad_diff = grad_diff + ((gx - gy) ** 2).sum()
                grad_diff.backward()
                return grad_diff
            
            #optimizer.step(closure)
            current_loss = optimizer.step(closure).item()
            if np.isfinite(current_loss):
                current_mse = torch.mean((torch.sigmoid(dummy_data) - gt_data) ** 2).item()
                if current_loss < best_loss_value:
                    best_loss_value = current_loss
                    best_dummy = torch.sigmoid(dummy_data).detach().clone()
                    best_mse_value = current_mse


            # ---- EARLY STOPPING (configurable from main) ----
            if not np.isfinite(current_loss):
                nan_count += 1
                early_stop_reason = "nan_or_inf"
                early_stop_iter = iters
                print(f"[GPU {device_id}] Early stop ({method}): NaN/Inf at iter {iters}")
                if nan_count >= max_nan:
                    break
            else:
                # update best / plateau counter using RELATIVE improvement
                if best_loss == float("inf"):
                    best_loss = current_loss
                    no_improve = 0
                else:
                    rel_improve = (best_loss - current_loss) / max(abs(best_loss), 1e-12)
                    if rel_improve > min_rel_improve:
                        best_loss = current_loss
                        no_improve = 0
                    elif iters >= warmup:
                        no_improve += 1

            # convergence
            if iters >= warmup and current_loss < loss_tol:
                early_stop_reason = "loss_tol"
                early_stop_iter = iters
                print(f"[GPU {device_id}] Early stop ({method}): loss_tol reached at iter {iters} (loss={current_loss:.3e})")
                break

            # explosion check
            if iters >= warmup and best_loss < float("inf") and current_loss > explode_factor * best_loss:
                early_stop_reason = "explosion"
                early_stop_iter = iters
                print(f"[GPU {device_id}] Early stop ({method}): exploded at iter {iters} (loss={current_loss:.3e}, best={best_loss:.3e})")
                break

            # plateau check
            if iters >= warmup and no_improve >= patience:
                early_stop_reason = "plateau"
                early_stop_iter = iters
                print(f"[GPU {device_id}] Early stop ({method}): plateau at iter {iters} (best={best_loss:.3e})")
                break
            # ---- END EARLY STOPPING ----

            losses.append(current_loss)
            #mses.append(torch.mean((torch.sigmoid(dummy_data) - gt_data) ** 2).item())
            current_mse = torch.mean((torch.sigmoid(dummy_data) - gt_data) ** 2).item()
            mses.append(current_mse)
            
            if iters % 100 == 0:
                print(f'[GPU {device_id}] iters {iters}, loss = {current_loss:.8f}, mse = {mses[-1]:.8f}')

            # if current_loss < loss_tol:
            #     break

        #final_recon[method] = torch.sigmoid(dummy_data).detach().clone()
        if best_dummy is not None:
            final_recon[method] = best_dummy
        else:
            final_recon[method] = torch.sigmoid(dummy_data).detach().clone()

        if method == 'iDLG':
            loss_iDLG = losses
            label_iDLG = label_pred.item()
            mse_iDLG = mses
        else:
            loss_iDLG_masked = losses
            label_iDLG_masked = label_pred.item()
            mse_iDLG_masked = mses
    
        early_stop_reason_dict[method] = early_stop_reason
        early_stop_iter_dict[method] = early_stop_iter

    # Prepare results to send back
    result = {
        'idx_net': idx_net,
        'device_id': device_id,
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
        'early_stop_reason': early_stop_reason_dict,
        'early_stop_iter': early_stop_iter_dict,
    }
    
    print(f"[GPU {device_id}] putting result for experiment {idx_net}", flush=True)
    result_queue.put(result)
    print(f"[GPU {device_id}] finished put for experiment {idx_net}", flush=True)