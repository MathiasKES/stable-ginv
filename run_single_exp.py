import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision
import os 
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "invertinggradients"))
# import inversefed
import consts

from Misc_functions import get_keep_ids, compute_psnr_from_mse, build_network, get_keep_ids_by_gradsize, get_entry_masks_by_gradsize, get_prefix_keep_ids, compute_jacobian_rank
from Network import weights_init

def run_single_experiment(idx_net, device_id, dst, dataset_name, config, result_queue):
    """Runs a single experiment on an assigned GPU"""
    torch.cuda.set_device(device_id)
    device = f'cuda:{device_id}'
    
    # Unpack config
    channel = config['channel']
    num_classes = config['num_classes']
    shape_img = config['shape_img']
    lr = config['lr']
    num_dummy = config['num_dummy']
    Iteration = config['Iteration']
    MASK_MODE = config['MASK_MODE']
    PREFIXES = config.get('PREFIXES', ())
    GRADSIZE_TOPK = config['GRADSIZE_TOPK']
    GRADSIZE_TOPFRAC = config['GRADSIZE_TOPFRAC']
    GRADSIZE_THRESHOLD = config['GRADSIZE_THRESHOLD']
    GRADSIZE_METRIC = config['GRADSIZE_METRIC']
    NETWORK_NAME = config['NETWORK_NAME']
    METHODS = config.get('METHODS', 'both')
    COMPUTE_JACOBIAN_RANK = config.get('COMPUTE_JACOBIAN_RANK', False)

    seed = config.get("run_id", 0) + idx_net + 1 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    net = build_network(NETWORK_NAME, channel=channel, num_classes=num_classes, input_size=shape_img)
    if NETWORK_NAME not in ["resnet20", "resnet18"]:
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

    if METHODS == "idlg":
        methods_to_run = ["iDLG"]
    elif METHODS == "masked":
        methods_to_run = ["iDLG_masked"]
    else:
        methods_to_run = ["iDLG", "iDLG_masked"]

    jac_rank_iDLG = None
    jac_rank_iDLG_masked = None
    jac_shape_iDLG = None
    jac_shape_iDLG_masked = None

    for method in methods_to_run:
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

        candidate_ids = None
        # choose which gradient tensors are "shared"
        keep_ids = None
        entry_masks = None
        ranked = None

        if method == "iDLG":
            keep_ids = get_keep_ids("all", net=net)
        else:
            if MASK_MODE.startswith("prefix_"):
                candidate_ids = get_prefix_keep_ids(net, PREFIXES)

            if MASK_MODE == "gradsize_topk":
                keep_ids, ranked = get_keep_ids_by_gradsize(
                    original_dy_dx, mode="topk", topk=GRADSIZE_TOPK,
                    metric=GRADSIZE_METRIC)

            elif MASK_MODE == "gradsize_topfrac":
                keep_ids, ranked = get_keep_ids_by_gradsize(
                    original_dy_dx, mode="topfrac", top_frac=GRADSIZE_TOPFRAC,
                    metric=GRADSIZE_METRIC)

            elif MASK_MODE == "gradsize_topk_entries":
                entry_masks, observed_entries, total_entries = get_entry_masks_by_gradsize(
                    original_dy_dx, mode="topk_entries", topk=GRADSIZE_TOPK)

            elif MASK_MODE == "gradsize_topfrac_entries":
                entry_masks, observed_entries, total_entries = get_entry_masks_by_gradsize(
                    original_dy_dx, mode="topfrac_entries", top_frac=GRADSIZE_TOPFRAC)

            elif MASK_MODE == "prefix_topk":
                keep_ids, ranked = get_keep_ids_by_gradsize(
                    original_dy_dx, mode="topk", topk=GRADSIZE_TOPK,
                    metric=GRADSIZE_METRIC, candidate_ids=candidate_ids)

            elif MASK_MODE == "prefix_topfrac":
                keep_ids, ranked = get_keep_ids_by_gradsize(
                    original_dy_dx, mode="topfrac", top_frac=GRADSIZE_TOPFRAC,
                    metric=GRADSIZE_METRIC, candidate_ids=candidate_ids)

            elif MASK_MODE == "prefix_topk_entries":
                entry_masks, observed_entries, total_entries = get_entry_masks_by_gradsize(
                    original_dy_dx, mode="topk_entries", topk=GRADSIZE_TOPK,
                    candidate_ids=candidate_ids)

            elif MASK_MODE == "prefix_topfrac_entries":
                entry_masks, observed_entries, total_entries = get_entry_masks_by_gradsize(
                    original_dy_dx, mode="topfrac_entries", top_frac=GRADSIZE_TOPFRAC,
                    candidate_ids=candidate_ids)

            elif MASK_MODE == "gradsize_threshold":
                keep_ids, ranked = get_keep_ids_by_gradsize(
                    original_dy_dx, mode="threshold", threshold=GRADSIZE_THRESHOLD,
                    metric=GRADSIZE_METRIC)

            elif MASK_MODE == "prefix":
                keep_ids = get_keep_ids(mask_mode="prefix", net=net, prefixes=PREFIXES)

            elif MASK_MODE == "resnet_l1_fc":
                keep_ids = get_keep_ids(mask_mode="prefix", net=net, prefixes=("layer1", "linear"))

            else:
                keep_ids = get_keep_ids(MASK_MODE)   

        unknowns = int(gt_data[0].numel())

        if entry_masks is not None:
            observed_entries = sum(
                int(m.sum().item()) for m in entry_masks if m is not None
            )
            total_entries = sum(g.numel() for g in original_dy_dx if g is not None)
        else:
            observed_entries = sum(
                g.numel() for i, g in enumerate(original_dy_dx)
                if g is not None and i in keep_ids
            )
            total_entries = sum(g.numel() for g in original_dy_dx if g is not None)

        kept_fraction = observed_entries / total_entries

        jacobian_rank = None
        jacobian_shape = None

        print(f"[GPU {device_id}] {method}: observed_entries={observed_entries}, "
            f"total_entries={total_entries}, kept_fraction={kept_fraction:.4f}, "
            f"unknowns={unknowns}")

        if COMPUTE_JACOBIAN_RANK:
            if num_dummy != 1:
                raise ValueError("Jacobian-rank computation currently assumes num_dummy=1.")

            jacobian_rank, jacobian_shape, jac_obs, jac_unknowns = compute_jacobian_rank(
                net=net,
                x_norm=gt_data_norm,
                y=gt_label,
                criterion=criterion,
                keep_ids=keep_ids,
                entry_masks=entry_masks,
            )

            print(f"[GPU {device_id}] {method}: jacobian_shape={jacobian_shape}, "
                f"jacobian_rank={jacobian_rank}, unknowns={jac_unknowns}")

            if jacobian_rank < jac_unknowns:
                print(f"[GPU {device_id}] {method}: Jacobian rank too small for unique local reconstruction "
                    f"({jacobian_rank} < {jac_unknowns})")
        
        if observed_entries < unknowns:
            print(f"[GPU {device_id}] {method}: too few gradients for reconstruction "
                f"({observed_entries} < {unknowns})")

            final_recon[method] = torch.zeros_like(gt_data)

            if method == 'iDLG':
                loss_iDLG = [float("inf")]
                label_iDLG = label_pred.item()
                mse_iDLG = [float("inf")]
            else:
                loss_iDLG_masked = [float("inf")]
                label_iDLG_masked = label_pred.item()
                mse_iDLG_masked = [float("inf")]

            early_stop_reason_dict[method] = "too_few_gradients"
            early_stop_iter_dict[method] = 0
            continue
        
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
                    if entry_masks is not None:
                        m = entry_masks[i]
                        if m is None:
                            continue
                        diff = gx[m] - gy[m]
                        grad_diff = grad_diff + (diff ** 2).sum()
                    else:
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
            jac_rank_iDLG = jacobian_rank
            jac_shape_iDLG = jacobian_shape
        else:
            loss_iDLG_masked = losses
            label_iDLG_masked = label_pred.item()
            mse_iDLG_masked = mses
            jac_rank_iDLG_masked = jacobian_rank
            jac_shape_iDLG_masked = jacobian_shape
    
        early_stop_reason_dict[method] = early_stop_reason
        early_stop_iter_dict[method] = early_stop_iter

    # Prepare results to send back
    result = {
    'idx_net': idx_net,
    'device_id': device_id,
    'gt_data': gt_data.detach().cpu().numpy(),
    'final_recon': {k: v.detach().cpu().numpy() for k, v in final_recon.items()},
    'psnr_idlg': compute_psnr_from_mse(mse_iDLG[-1], max_val=1.0) if 'iDLG' in final_recon else None,
    'psnr_masked': compute_psnr_from_mse(mse_iDLG_masked[-1], max_val=1.0) if 'iDLG_masked' in final_recon else None,
    'loss_iDLG': loss_iDLG[-1] if 'iDLG' in final_recon else None,
    'mse_iDLG': mse_iDLG[-1] if 'iDLG' in final_recon else None,
    'loss_iDLG_masked': loss_iDLG_masked[-1] if 'iDLG_masked' in final_recon else None,
    'mse_iDLG_masked': mse_iDLG_masked[-1] if 'iDLG_masked' in final_recon else None,
    'label_iDLG': label_iDLG if 'iDLG' in final_recon else None,
    'label_iDLG_masked': label_iDLG_masked if 'iDLG_masked' in final_recon else None,
    'jac_rank_iDLG': jac_rank_iDLG if 'iDLG' in final_recon else None,
    'jac_shape_iDLG': jac_shape_iDLG if 'iDLG' in final_recon else None,
    'jac_rank_iDLG_masked': jac_rank_iDLG_masked if 'iDLG_masked' in final_recon else None,
    'jac_shape_iDLG_masked': jac_shape_iDLG_masked if 'iDLG_masked' in final_recon else None,
    'gt_label': gt_label.detach().cpu().numpy(),
    'imidx_list': imidx_list,
    'early_stop_reason': early_stop_reason_dict,
    'early_stop_iter': early_stop_iter_dict,
}
    
    print(f"[GPU {device_id}] putting result for experiment {idx_net}", flush=True)
    result_queue.put(result)
    print(f"[GPU {device_id}] finished put for experiment {idx_net}", flush=True)