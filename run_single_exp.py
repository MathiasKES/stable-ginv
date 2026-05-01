# run_single_exp.py
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision
import os 
import sys
import consts # Local file
#from skimage.metrics import structural_similarity as ssim

from Misc_functions import (get_keep_ids, compute_psnr_from_mse, build_network, get_keep_ids_by_gradsize, 
get_entry_masks_by_gradsize, get_prefix_keep_ids, compute_jacobian_rank, total_variation, get_entry_masks_by_prefix_group, get_keep_ids_by_prefix_group, compute_grad_match_loss, make_scheduler)
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
    PREFIX_LAYER_FRACS = config.get('PREFIX_LAYER_FRACS', {})
    GRADSIZE_TOPK = config['GRADSIZE_TOPK']
    GRADSIZE_TOPFRAC = config['GRADSIZE_TOPFRAC']
    GRADSIZE_THRESHOLD = config['GRADSIZE_THRESHOLD']
    GRADSIZE_METRIC = config['GRADSIZE_METRIC']
    NETWORK_NAME = config['NETWORK_NAME']
    NETWORK_TRAINED = config['NETWORK_TRAINED']
    METHODS = config.get('METHODS', 'both')
    COMPUTE_JACOBIAN_RANK = config.get('COMPUTE_JACOBIAN_RANK', False)
    JACOBIAN_MAX_ENTRIES = config.get('JACOBIAN_MAX_ENTRIES', 4000)
    JACOBIAN_SELECT_MODE = config.get('JACOBIAN_SELECT_MODE', 'topk_abs')
    TV_WEIGHT = config.get('TV_WEIGHT', 0.0)
    OPTIMIZER = config.get('OPTIMIZER', 'lbfgs')
    NUM_RESTARTS = config.get('NUM_RESTARTS', 1)
    MAX_ITERATION = config.get('MAX_ITERATION',20)
    HISTORY_SIZE = config.get('HISTORY_SIZE',100)
    SAVE_GIF = config.get('SAVE_GIF', False)
    FRAME_INTERVAL = config.get('FRAME_INTERVAL', 20)
    GRAD_LOSS = config.get('GRAD_LOSS', 'cos').lower() # options: "l2" or "cos"

    seed = config.get("run_id", 0) + idx_net + 1 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    OPTIMIZE_NORM_SPACE = NETWORK_TRAINED

    net = build_network(NETWORK_NAME, channel=channel, num_classes=num_classes, input_size=shape_img, pretrained=NETWORK_TRAINED)
    if NETWORK_TRAINED:
        print(f"[GPU {device_id}] Loaded ImageNet-pretrained weights for {NETWORK_NAME}")
    elif NETWORK_NAME in ["LeNet", "LeNet_bigger", "MediumCNN", "BiggerCNN"]:
            net.apply(weights_init)

    net = net.to(device)
    net.eval()

    if idx_net == 0 and device_id == 0:
        # for i, (name, param) in enumerate(net.named_parameters()):
        #     print(i, name, tuple(param.shape))
        print(f'[GPU {device_id}] Running {idx_net} experiment')
    
    idx_shuffle = np.random.permutation(len(dst))
    tt = transforms.Compose([transforms.ToTensor()])
    tp = transforms.Compose([transforms.ToPILImage()])

    final_recon = {}
    early_stop_reason_dict = {}
    early_stop_iter_dict = {}

    best_loss_iDLG = None
    best_mse_iDLG = None
    best_loss_iDLG_masked = None
    best_mse_iDLG_masked = None

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

    init_frames_by_method = {}
    recon_frames_by_method = {}

    for method in methods_to_run:
        #print(f'[GPU {device_id}] {method}, Try to generate {num_dummy} images')

        best_restart_loss_value = float("inf")
        best_restart_mse_value = None
        best_restart_dummy = None
        best_restart_losses = None
        best_restart_mses = None
        best_restart_early_stop_reason = None
        best_restart_early_stop_iter = None

        if SAVE_GIF:
            _best_restart_init_np = None
            _best_restart_frames = []

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

        if NETWORK_TRAINED and channel == 3:
            dm = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, channel, 1, 1)
            ds = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, channel, 1, 1)
        else:
            dm = torch.tensor(getattr(consts, f'{dataset_name.lower()}_mean'), device=device).view(1, channel, 1, 1)
            ds = torch.tensor(getattr(consts, f'{dataset_name.lower()}_std'), device=device).view(1, channel, 1, 1)

        lower_bound = -dm / ds
        upper_bound = (1.0 - dm) / ds

        # ---- compute original gradients ----
        gt_data_norm = (gt_data - dm) / ds # normalize gt data
        out = net(gt_data_norm)
        y = criterion(out, gt_label)
        dy_dx = torch.autograd.grad(y, net.parameters())
        original_dy_dx = [g.detach().clone() for g in dy_dx]

        # iDLG label inference
        #label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape((1,))

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
                keep_ids, ranked = get_keep_ids_by_prefix_group(
                    net=net,
                    original_dy_dx=original_dy_dx,
                    prefixes=PREFIXES,
                    mode="topk",
                    topk=GRADSIZE_TOPK,
                    prefix_top_ks={k: int(v) for k, v in PREFIX_LAYER_FRACS.items()},
                    metric=GRADSIZE_METRIC,
                )
            elif MASK_MODE == "prefix_topfrac":
                keep_ids, ranked = get_keep_ids_by_prefix_group(
                    net=net,
                    original_dy_dx=original_dy_dx,
                    prefixes=PREFIXES,
                    mode="topfrac",
                    top_frac=GRADSIZE_TOPFRAC,
                    prefix_top_fracs=PREFIX_LAYER_FRACS,
                    metric=GRADSIZE_METRIC,
                )
            
            elif MASK_MODE == "prefix_topk_entries":
                entry_masks, observed_entries, total_entries = get_entry_masks_by_gradsize(
                    original_dy_dx, mode="topk_entries", topk=GRADSIZE_TOPK,
                    candidate_ids=candidate_ids)

            elif MASK_MODE == "prefix_topfrac_entries":
                entry_masks, observed_entries, total_entries = get_entry_masks_by_gradsize(
                    original_dy_dx, mode="topfrac_entries", top_frac=GRADSIZE_TOPFRAC,
                    candidate_ids=candidate_ids)

            elif MASK_MODE == "prefix_topk_entries_layer":
                entry_masks, observed_entries, total_entries = get_entry_masks_by_prefix_group(
                    net=net,
                    original_dy_dx=original_dy_dx,
                    prefixes=PREFIXES,
                    mode="topk_entries",
                    topk=GRADSIZE_TOPK,
                )

            elif MASK_MODE == "prefix_topfrac_entries_layer":
                entry_masks, observed_entries, total_entries = get_entry_masks_by_prefix_group(
                    net=net,
                    original_dy_dx=original_dy_dx,
                    prefixes=PREFIXES,
                    mode="topfrac_entries",
                    top_frac=GRADSIZE_TOPFRAC,
                    prefix_top_fracs=PREFIX_LAYER_FRACS,
                )
            elif MASK_MODE == "gradsize_threshold":
                keep_ids, ranked = get_keep_ids_by_gradsize(
                    original_dy_dx, mode="threshold", threshold=GRADSIZE_THRESHOLD,
                    metric=GRADSIZE_METRIC)

            elif MASK_MODE == "prefix":
                keep_ids = get_keep_ids(mask_mode="prefix", net=net, prefixes=PREFIXES)

            else:
                keep_ids = get_keep_ids(MASK_MODE)   

        final_weight_idx = len(original_dy_dx) - 2

        label_pred = None
        label_inference_available = False

        if method == "iDLG":
            label_pred = torch.argmin(torch.sum(original_dy_dx[final_weight_idx], dim=-1), dim=-1).detach().reshape((1,))
            label_inference_available = True
        else:
            if entry_masks is not None:
                m = entry_masks[final_weight_idx]
                if m is not None and bool(m.all()):
                    label_pred = torch.argmin(torch.sum(original_dy_dx[final_weight_idx], dim=-1), dim=-1).detach().reshape((1,))
                    label_inference_available = True
            else:
                if final_weight_idx in keep_ids:
                    label_pred = torch.argmin(torch.sum(original_dy_dx[final_weight_idx], dim=-1), dim=-1).detach().reshape((1,))
                    label_inference_available = True

        if not label_inference_available:
            print(f"[GPU {device_id}] {method}: label inference unavailable under current mask")

            final_recon[method] = torch.zeros_like(gt_data)

            if method == 'iDLG':
                loss_iDLG = [float("inf")]
                label_iDLG = None
                mse_iDLG = [float("inf")]
                best_loss_iDLG = float("inf")
                best_mse_iDLG = float("inf")
            else:
                loss_iDLG_masked = [float("inf")]
                label_iDLG_masked = None
                mse_iDLG_masked = [float("inf")]
                best_loss_iDLG_masked = float("inf")
                best_mse_iDLG_masked = float("inf")

            early_stop_reason_dict[method] = "label_inference_unavailable"
            early_stop_iter_dict[method] = 0
            continue

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

        if idx_net == 0 and device_id == 0:
            print(f"[GPU {device_id}] {method}: observed_entries={observed_entries}, "
                f"total_entries={total_entries}, kept_fraction={kept_fraction:.4f}, "
                f"unknowns={unknowns}")
            
        if MASK_MODE in ["prefix_topfrac_entries_layer", "prefix_topk_entries_layer"] and entry_masks is not None:
            print(f"[GPU {device_id}] kept entries per prefix:")
            for prefix in PREFIXES:
                kept = 0
                total = 0
                for i, (name, _) in enumerate(net.named_parameters()):
                    if (name == prefix or name.startswith(prefix + ".")) and entry_masks[i] is not None:
                        kept += int(entry_masks[i].sum().item())
                        total += entry_masks[i].numel()
                if total > 0:
                    print(f"  {prefix}: kept {kept}/{total} = {kept/total:.4f}")
        
        if MASK_MODE in ["prefix_topfrac", "prefix_topk"] and keep_ids is not None:
            print(f"[GPU {device_id}] kept tensors per prefix:")
            named_params = list(net.named_parameters())
            keep_ids_set = set(keep_ids)

            for prefix in PREFIXES:
                total = 0
                kept = 0
                for i, (name, _) in enumerate(named_params):
                    if name == prefix or name.startswith(prefix + "."):
                        total += 1
                        if i in keep_ids_set:
                            kept += 1

                if total > 0:
                    if MASK_MODE == "prefix_topfrac":
                        req = PREFIX_LAYER_FRACS.get(prefix, GRADSIZE_TOPFRAC)
                    else:
                        req = int(PREFIX_LAYER_FRACS.get(prefix, GRADSIZE_TOPK))
                    # req = PREFIX_LAYER_FRACS.get(prefix, GRADSIZE_TOPFRAC) if MASK_MODE == "prefix_topfrac" else GRADSIZE_TOPK
                    print(f"  {prefix}: kept {kept}/{total} = {kept/total:.4f}, requested={req}")
                            
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
                max_entries=JACOBIAN_MAX_ENTRIES,
                select_mode=JACOBIAN_SELECT_MODE,
                device_for_J="cpu",
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
                best_loss_iDLG = float("inf")
                best_mse_iDLG = float("inf")
            else:
                loss_iDLG_masked = [float("inf")]
                label_iDLG_masked = label_pred.item()
                mse_iDLG_masked = [float("inf")]
                best_loss_iDLG_masked = float("inf")
                best_mse_iDLG_masked = float("inf")

            early_stop_reason_dict[method] = "too_few_gradients"
            early_stop_iter_dict[method] = 0
            continue

        all_params = list(net.parameters())
        if entry_masks is not None:
            selected_ids = [
                i for i, m in enumerate(entry_masks)
                if m is not None and m.any()
            ]
        else:
            selected_ids = sorted(list(keep_ids))

        selected_params = [all_params[i] for i in selected_ids]
        selected_original = [original_dy_dx[i] for i in selected_ids]
        selected_entry_masks = [entry_masks[i] for i in selected_ids] if entry_masks is not None else None

        for restart_idx in range(NUM_RESTARTS):
            restart_seed = seed * 1000 + restart_idx
            torch.manual_seed(restart_seed)
            np.random.seed(restart_seed)
            torch.cuda.manual_seed_all(restart_seed)

            print(f"[GPU {device_id}] {method}: restart {restart_idx+1}/{NUM_RESTARTS}")

            #dummy_data = (torch.randn(gt_data.size(), device=device)).requires_grad_(True)
            #dummy_data = torch.rand(gt_data.size(), device=device, requires_grad=True)
            if OPTIMIZE_NORM_SPACE:
                dummy_raw_init = torch.rand(gt_data.size(), device=device)
                dummy_data = ((dummy_raw_init - dm) / ds).detach().requires_grad_(True)
            else:
                dummy_data = torch.rand(gt_data.size(), device=device, requires_grad=True)

            if SAVE_GIF:
                #_restart_init_np = torch.sigfmoid(dummy_data).detach().cpu().numpy()
                #_restart_init_np = dummy_data.detach().cpu().numpy()
                if OPTIMIZE_NORM_SPACE:
                    _restart_init_np = (dummy_data.detach() * ds + dm).clamp(0.0, 1.0).cpu().numpy()
                else:
                    _restart_init_np = dummy_data.detach().cpu().numpy()
                _restart_frames = []
                _last_gif_iter = -1

            scheduler = None

            if OPTIMIZER == "lbfgs":
                optimizer = torch.optim.LBFGS(
                    [dummy_data],
                    lr=lr,
                    max_iter=MAX_ITERATION,
                    history_size=HISTORY_SIZE,
                )
                phase = "lbfgs"

            elif OPTIMIZER in ["adam", "signed_adam"]:
                optimizer = torch.optim.Adam([dummy_data], lr=lr)
                scheduler = make_scheduler(optimizer, Iteration)
                phase = OPTIMIZER

            elif OPTIMIZER in ["adamw", "signed_adamw", "adamw_lbfgs"]:
                optimizer = torch.optim.AdamW([dummy_data], lr=lr, weight_decay=1e-5)
                scheduler = make_scheduler(optimizer, Iteration)
                phase = "adamw" if OPTIMIZER == "adamw_lbfgs" else OPTIMIZER

            else:
                raise ValueError(f"Unknown optimizer: {OPTIMIZER}")

            losses = []
            mses = []

            # es = config.get("EarlyStop", {})
            # best_loss = float("inf")
            # no_improve = 0

            # patience = int(es.get("patience", 100))
            # min_rel_improve = float(es.get("min_rel_improve", 1e-6))
            # explode_factor = float(es.get("explode_factor", 30.0))
            # warmup = int(es.get("warmup", 100))
            # loss_tol = float(es.get("loss_tol", 1e-6))
            # max_nan = int(es.get("max_nan", 1))

            # nan_count = 0
            # early_stop_reason = None
            # early_stop_iter = None
            # best_loss_value = float("inf")
            # best_dummy = None
            # best_mse_value = None

            early_stop_reason = "fixed_iterations"
            early_stop_iter = Iteration
            best_loss_value = float("inf")
            best_dummy = None
            best_mse_value = None

            for iters in range(Iteration):
                if phase == "lbfgs":
                    def closure():
                        optimizer.zero_grad()

                        #x = torch.sigmoid(dummy_data)
                        # x = dummy_data
                        # x_norm = (x - dm) / ds

                        if OPTIMIZE_NORM_SPACE:
                            x_norm = dummy_data
                            x_raw = (dummy_data * ds + dm).clamp(0.0, 1.0)
                        else:
                            x_raw = dummy_data
                            x_norm = (x_raw - dm) / ds

                        pred = net(x_norm)
                        dummy_loss = criterion(pred, label_pred)
                        dummy_dy_dx = torch.autograd.grad(dummy_loss, selected_params, create_graph=True)

                        grad_diff, _ = compute_grad_match_loss(dummy_dy_dx, selected_original, selected_entry_masks=selected_entry_masks, grad_loss=GRAD_LOSS)

                        #tv_loss = total_variation(x)
                        tv_loss = total_variation(x_raw)
                        total_loss = grad_diff + TV_WEIGHT * tv_loss
                        total_loss.backward()
                        return total_loss

                    current_loss = optimizer.step(closure).item()
                    with torch.no_grad():
                        #dummy_data.clamp_(0.0, 1.0)
                        if OPTIMIZE_NORM_SPACE:
                            dummy_data.clamp_(lower_bound, upper_bound)
                        else:
                            dummy_data.clamp_(0.0, 1.0)

                elif phase in ["adam", "adamw", "signed_adam", "signed_adamw"]:
                    optimizer.zero_grad()

                    #x = torch.sigmoid(dummy_data)
                    # x = dummy_data
                    # x_norm = (x - dm) / ds

                    if OPTIMIZE_NORM_SPACE:
                        x_norm = dummy_data
                        x_raw = (dummy_data * ds + dm).clamp(0.0, 1.0)
                    else:
                        x_raw = dummy_data
                        x_norm = (x_raw - dm) / ds

                    pred = net(x_norm)
                    dummy_loss = criterion(pred, label_pred)
                    dummy_dy_dx = torch.autograd.grad(dummy_loss, selected_params, create_graph=True)

                    grad_diff, num_terms = compute_grad_match_loss(dummy_dy_dx, selected_original, selected_entry_masks=selected_entry_masks, grad_loss=GRAD_LOSS)
                    
                    #tv_loss = total_variation(x)
                    tv_loss = total_variation(x_raw)
                    total_loss = grad_diff + TV_WEIGHT * tv_loss
                    total_loss.backward()

                    if phase in ["signed_adam", "signed_adamw"] and dummy_data.grad is not None:
                        with torch.no_grad():
                            dummy_data.grad.sign_()

                    optimizer.step()
                    with torch.no_grad():
                        #dummy_data.clamp_(0.0, 1.0)
                        if OPTIMIZE_NORM_SPACE:
                            dummy_data.clamp_(lower_bound, upper_bound)
                        else:
                            dummy_data.clamp_(0.0, 1.0)

                    if scheduler is not None:
                        scheduler.step()

                    current_loss = total_loss.item()

                else:
                    raise ValueError(f"Unknown optimizer: {OPTIMIZER}")

                #current_x = torch.sigmoid(dummy_data).detach().clone()
                # current_x = dummy_data.detach().clone()
                if OPTIMIZE_NORM_SPACE:
                    current_x = (dummy_data.detach() * ds + dm).clamp(0.0, 1.0)
                else:
                    current_x = dummy_data.detach().clone()
                
                current_mse = torch.mean((current_x - gt_data) ** 2).item()

                if SAVE_GIF and iters % FRAME_INTERVAL == 0:
                    _restart_frames.append({
                        'iter': iters,
                        'dummy': current_x.cpu().numpy(),
                        'loss': current_loss if np.isfinite(current_loss) else float('inf'),
                        'mse': current_mse,
                    })
                    _last_gif_iter = iters

                if np.isfinite(current_loss):
                    if current_loss < best_loss_value:
                        best_loss_value = current_loss
                        best_mse_value = current_mse
                        best_dummy = current_x.detach().clone()

                # if not np.isfinite(current_loss):
                #     nan_count += 1
                #     early_stop_reason = "nan_or_inf"
                #     early_stop_iter = iters
                #     print(f"[GPU {device_id}] Early stop ({method}, restart {restart_idx+1}): NaN/Inf at iter {iters}")
                #     if nan_count >= max_nan:
                #         break
                # else:
                #     if best_loss == float("inf"):
                #         best_loss = current_loss
                #         no_improve = 0
                #     else:
                #         rel_improve = (best_loss - current_loss) / max(abs(best_loss), 1e-12)
                #         if rel_improve > min_rel_improve:
                #             best_loss = current_loss
                #             no_improve = 0
                #         elif iters >= warmup:
                #             no_improve += 1

                # if iters >= warmup and current_loss < loss_tol:
                #     early_stop_reason = "loss_tol"
                #     early_stop_iter = iters
                #     print(f"[GPU {device_id}] Early stop ({method}, restart {restart_idx+1}): loss_tol reached at iter {iters} (loss={current_loss:.3e})")
                #     break

                # if iters >= warmup and best_loss < float("inf") and current_loss > explode_factor * best_loss:
                #     early_stop_reason = "explosion"
                #     early_stop_iter = iters
                #     print(f"[GPU {device_id}] Early stop ({method}, restart {restart_idx+1}): exploded at iter {iters} (loss={current_loss:.3e}, best={best_loss:.3e})")
                #     break

                # if iters >= warmup and no_improve >= patience:
                #     if OPTIMIZER == "adamw_lbfgs" and phase == "adamw":
                #         print(f"[GPU {device_id}] Switching AdamW -> L-BFGS at iter {iters} (best={best_loss:.3e})")
                #         scheduler = None
                #         optimizer = torch.optim.LBFGS([dummy_data], lr=1, max_iter=MAX_ITERATION, history_size=HISTORY_SIZE)
                #         phase = "lbfgs"
                #         no_improve = 0
                #         best_loss = float("inf")
                #     else:
                #         early_stop_reason = "plateau"
                #         early_stop_iter = iters
                #         print(f"[GPU {device_id}] Early stop ({method}, restart {restart_idx+1}): plateau at iter {iters} (best={best_loss:.3e})")
                #         break

                losses.append(current_loss)
                mses.append(current_mse)

                if iters % 1000 == 0:
                    current_lr = optimizer.param_groups[0]["lr"]
                    print(f'[GPU {device_id}] {OPTIMIZER}({phase}) restart {restart_idx+1} iters {iters}, lr = {current_lr:.6g}, loss = {current_loss:.8f}, mse = {current_mse:.8f}')

            if SAVE_GIF and Iteration > 0 and _last_gif_iter != iters:
                #_final_x = torch.sigmoid(dummy_data).detach()
                #_final_x = dummy_data.detach()
                if OPTIMIZE_NORM_SPACE:
                    _final_x = (dummy_data.detach() * ds + dm).clamp(0.0, 1.0)
                else:
                    _final_x = dummy_data.detach().clone()
                _restart_frames.append({
                    'iter': iters,
                    'dummy': _final_x.cpu().numpy(),
                    'loss': losses[-1] if losses else float('inf'),
                    'mse': torch.mean((_final_x - gt_data) ** 2).item(),
                })

            # if best_mse_value is not None:
            #     if best_restart_mse_value is None or best_mse_value < best_restart_mse_value:
            #         best_restart_mse_value = best_mse_value
            #         best_restart_loss_value = best_loss_value
            #         #best_restart_dummy = best_dummy.clone() if best_dummy is not None else torch.sigmoid(dummy_data).detach().clone()
            #         best_restart_dummy = best_dummy.clone() if best_dummy is not None else dummy_data.detach().clone()
            #         best_restart_losses = losses[:]
            #         best_restart_mses = mses[:]
            #         best_restart_early_stop_reason = early_stop_reason
            #         best_restart_early_stop_iter = early_stop_iter
            #         if SAVE_GIF:
            #             _best_restart_init_np = _restart_init_np
            #             _best_restart_frames = _restart_frames[:]

            if best_dummy is not None:
                if best_restart_losses is None or best_loss_value < best_restart_loss_value:
                    best_restart_mse_value = best_mse_value
                    best_restart_loss_value = best_loss_value
                    best_restart_dummy = best_dummy.clone()
                    best_restart_losses = losses[:]
                    best_restart_mses = mses[:]
                    best_restart_early_stop_reason = early_stop_reason
                    best_restart_early_stop_iter = early_stop_iter
                    if SAVE_GIF:
                        _best_restart_init_np = _restart_init_np
                        _best_restart_frames = _restart_frames[:]

        if SAVE_GIF:
            init_frames_by_method[method] = _best_restart_init_np
            recon_frames_by_method[method] = _best_restart_frames

        if best_restart_dummy is not None:
            final_recon[method] = best_restart_dummy
        else:
            final_recon[method] = torch.zeros_like(gt_data)

        if method == 'iDLG':
            loss_iDLG = best_restart_losses if best_restart_losses is not None else [float("inf")]
            label_iDLG = label_pred.item()
            mse_iDLG = best_restart_mses if best_restart_mses is not None else [float("inf")]
            best_loss_iDLG = best_restart_loss_value
            best_mse_iDLG = best_restart_mse_value
            jac_rank_iDLG = jacobian_rank
            jac_shape_iDLG = jacobian_shape
        else:
            loss_iDLG_masked = best_restart_losses if best_restart_losses is not None else [float("inf")]
            label_iDLG_masked = label_pred.item()
            mse_iDLG_masked = best_restart_mses if best_restart_mses is not None else [float("inf")]
            best_loss_iDLG_masked = best_restart_loss_value
            best_mse_iDLG_masked = best_restart_mse_value
            jac_rank_iDLG_masked = jacobian_rank
            jac_shape_iDLG_masked = jacobian_shape

        early_stop_reason_dict[method] = best_restart_early_stop_reason
        early_stop_iter_dict[method] = best_restart_early_stop_iter

    # Prepare results to send back
    result = {
        'idx_net': idx_net,
        'device_id': device_id,
        'gt_data': gt_data.detach().cpu().numpy(),
        'final_recon': {k: v.detach().cpu().numpy() for k, v in final_recon.items()},

        'last_psnr_idlg': compute_psnr_from_mse(mse_iDLG[-1], max_val=1.0) if 'iDLG' in final_recon else None,
        'last_psnr_masked': compute_psnr_from_mse(mse_iDLG_masked[-1], max_val=1.0) if 'iDLG_masked' in final_recon else None,
        'last_loss_iDLG': loss_iDLG[-1] if 'iDLG' in final_recon else None,
        'last_mse_iDLG': mse_iDLG[-1] if 'iDLG' in final_recon else None,
        'last_loss_iDLG_masked': loss_iDLG_masked[-1] if 'iDLG_masked' in final_recon else None,
        'last_mse_iDLG_masked': mse_iDLG_masked[-1] if 'iDLG_masked' in final_recon else None,

        'best_psnr_idlg': compute_psnr_from_mse(best_mse_iDLG, max_val=1.0) if best_mse_iDLG is not None else None,
        'best_psnr_masked': compute_psnr_from_mse(best_mse_iDLG_masked, max_val=1.0) if best_mse_iDLG_masked is not None else None,
        'best_loss_iDLG': best_loss_iDLG,
        'best_mse_iDLG': best_mse_iDLG,
        'best_loss_iDLG_masked': best_loss_iDLG_masked,
        'best_mse_iDLG_masked': best_mse_iDLG_masked,

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
        'init_frames': init_frames_by_method,
        'recon_frames': recon_frames_by_method,
    }
    
    # print(f"[GPU {device_id}] putting result for experiment {idx_net}", flush=True)
    result_queue.put(result)
    # print(f"[GPU {device_id}] finished put for experiment {idx_net}", flush=True)