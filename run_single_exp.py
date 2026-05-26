# run_single_exp.py
import traceback
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import functions.consts as consts

from functions.masking import build_gradient_mask, _get_last_fc_param_indices
from helper.metrics import (compute_psnr_from_mse, compute_jacobian_rank, total_variation,
    compute_grad_match_loss, compute_ssim_batch)
from helper.Network import get_model, weights_init
from helper.training_utils import make_scheduler

from functions.io_utils import setstdout

def run_single_experiment(idx_net, device_id, dst, dataset_name, config, result_queue):
    setstdout(path=config.get('out_path'))
    try:
        _run_inner(idx_net, device_id, dst, dataset_name, config, result_queue)
    except Exception as exc:
        result_queue.put({
            'error': str(exc),
            'traceback': traceback.format_exc(),
            'idx_net': idx_net,
            'device_id': device_id,
        })

def _run_inner(idx_net, device_id, dst, dataset_name, config, result_queue):
    torch.cuda.set_device(device_id)
    device = f'cuda:{device_id}'

    # Unpack config
    channel = config['channel']
    num_classes = config['num_classes']
    shape_img = config['shape_img']
    lr = config['lr']
    GAMMA = config['GAMMA']
    num_dummy = config['num_dummy']
    Iteration = config['Iteration']
    MASK_MODE = config['MASK_MODE']
    PREFIXES = config.get('PREFIXES', ())
    PREFIX_LAYER_FRACS = config.get('PREFIX_LAYER_FRACS', {})
    GRADSIZE_TOPK = config['GRADSIZE_TOPK']
    GRADSIZE_TOPFRAC = config['GRADSIZE_TOPFRAC']
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
    SINGLE_RESTART_IDX = config.get('SINGLE_RESTART_IDX')  # None = run all restarts
    MAX_ITERATION = config.get('MAX_ITERATION', 20)
    HISTORY_SIZE = config.get('HISTORY_SIZE', 100)
    SAVE_GIF = config.get('SAVE_GIF', False)
    FRAME_INTERVAL = config.get('FRAME_INTERVAL', 20)
    GRAD_LOSS = config.get('GRAD_LOSS', 'cos').lower()

    seed = config.get("run_id", 0) + idx_net + 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    net = get_model(NETWORK_NAME, channel=channel, num_classes=num_classes, input_size=shape_img, pretrained=NETWORK_TRAINED)
    if NETWORK_TRAINED:
        print(f"[GPU {device_id}] Loaded ImageNet-pretrained weights for {NETWORK_NAME}")
    elif NETWORK_NAME in ["LeNet", "LeNet_bigger", "MediumCNN", "BiggerCNN"]:
        net.apply(weights_init)

    net = net.to(device)
    net.eval()

    idx_shuffle = np.random.permutation(len(dst))
    tt = transforms.Compose([transforms.ToTensor()])

    criterion = nn.CrossEntropyLoss().to(device)
    imidx_list = []

    # Build GT batch — computed once, shared across all methods
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

    # LeNet/LeNet_bigger use Sigmoid activations designed for raw [0,1] inputs.
    # Applying dataset normalisation shifts inputs into ≈[-2, +2], saturating
    # the sigmoid and zeroing out conv gradients — making inversion impossible.
    # Skip normalisation for these architectures to match the original iDLG paper.
    _sigmoid_nets = {"LeNet", "LeNet_bigger"}
    if NETWORK_NAME in _sigmoid_nets:
        dm = torch.zeros(1, channel, 1, 1, device=device)
        ds = torch.ones(1, channel, 1, 1, device=device)
    elif NETWORK_TRAINED and channel == 3:
        dm = torch.tensor(consts.imagenet_mean, device=device).view(1, channel, 1, 1)
        ds = torch.tensor(consts.imagenet_std,  device=device).view(1, channel, 1, 1)
    else:
        dm = torch.tensor(getattr(consts, f'{dataset_name.lower()}_mean'), device=device).view(1, channel, 1, 1)
        ds = torch.tensor(getattr(consts, f'{dataset_name.lower()}_std'),  device=device).view(1, channel, 1, 1)

    lower_bound = -dm / ds
    upper_bound = (1.0 - dm) / ds
    gt_data_norm = (gt_data - dm) / ds

    out = net(gt_data_norm)
    y = criterion(out, gt_label)
    dy_dx = torch.autograd.grad(y, net.parameters())
    original_dy_dx = [g.detach().clone() for g in dy_dx]
    total_entries = sum(g.numel() for g in original_dy_dx)

    final_recon = {}
    early_stop_reason_dict = {}
    early_stop_iter_dict = {}

    # Per-method result accumulators (avoids duplicated if/else dispatch blocks)
    _losses = {}
    _labels = {}
    _mses = {}
    _best_loss = {}
    _best_mse = {}
    _best_ssim = {}
    _jac_rank = {}
    _jac_shape = {}
    _psnr_per_restart = {}
    _img_per_restart = {}
    _mse_per_restart = {}
    _ssim_per_restart = {}

    if METHODS == "idlg":
        methods_to_run = ["iDLG"]
    elif METHODS == "masked":
        methods_to_run = ["iDLG_masked"]
    else:
        methods_to_run = ["iDLG", "iDLG_masked"]

    init_frames_by_method = {}
    recon_frames_by_method = {}

    for method in methods_to_run:

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

        keep_ids, entry_masks = build_gradient_mask(
            method="idlg" if method == "iDLG" else "masked",
            mask_mode=MASK_MODE,
            net=net,
            original_dy_dx=original_dy_dx,
            prefixes=PREFIXES,
            prefix_layer_fracs=PREFIX_LAYER_FRACS,
            gradsize_topk=GRADSIZE_TOPK,
            gradsize_topfrac=GRADSIZE_TOPFRAC,
            gradsize_metric=GRADSIZE_METRIC,
        )

        if entry_masks is not None:
            observed_entries = sum(int(m.sum().item()) for m in entry_masks if m is not None)
        else:
            observed_entries = sum(
                g.numel() for i, g in enumerate(original_dy_dx)
                if g is not None and i in keep_ids
            )

        _named_params_list = list(net.named_parameters())
        _last_fc_ids = _get_last_fc_param_indices(net)
        final_weight_idx = next(
            i for i in sorted(_last_fc_ids) if _named_params_list[i][0].endswith('.weight')
        )

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
            elif keep_ids is not None:
                if final_weight_idx in keep_ids:
                    label_pred = torch.argmin(torch.sum(original_dy_dx[final_weight_idx], dim=-1), dim=-1).detach().reshape((1,))
                    label_inference_available = True

        if not label_inference_available:
            print(f"[GPU {device_id}] {method}: label inference unavailable under current mask")
            final_recon[method] = torch.zeros_like(gt_data)
            _losses[method] = [float("inf")]
            _labels[method] = None
            _mses[method] = [float("inf")]
            _best_loss[method] = float("inf")
            _best_mse[method] = float("inf")
            early_stop_reason_dict[method] = "label_inference_unavailable"
            early_stop_iter_dict[method] = 0
            continue

        unknowns = int(gt_data[0].numel())

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
            _losses[method] = [float("inf")]
            _labels[method] = label_pred.item()
            _mses[method] = [float("inf")]
            _best_loss[method] = float("inf")
            _best_mse[method] = float("inf")
            early_stop_reason_dict[method] = "too_few_gradients"
            early_stop_iter_dict[method] = 0
            continue

        all_params = list(net.parameters())
        if entry_masks is not None:
            selected_ids = [i for i, m in enumerate(entry_masks) if m is not None and m.any()]
        else:
            selected_ids = sorted(list(keep_ids))

        selected_params = [all_params[i] for i in selected_ids]
        selected_original = [original_dy_dx[i] for i in selected_ids]
        selected_entry_masks = [entry_masks[i] for i in selected_ids] if entry_masks is not None else None

        _best_mse_per_restart = []
        _best_img_per_restart = []
        _ssim_per_restart_list = []
        _restart_range = [SINGLE_RESTART_IDX] if SINGLE_RESTART_IDX is not None else range(NUM_RESTARTS)

        for restart_idx in _restart_range:
            restart_seed = seed * 1000 + restart_idx
            torch.manual_seed(restart_seed)
            np.random.seed(restart_seed)
            torch.cuda.manual_seed_all(restart_seed)

            print(f"[GPU {device_id}] {method}: restart {restart_idx+1}/{NUM_RESTARTS}")

            dummy_data = torch.randn(gt_data.size(), device=device).requires_grad_(True)

            if SAVE_GIF:
                _restart_init_np = (dummy_data.detach() * ds + dm).clamp(0.0, 1.0).cpu().numpy()
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
                scheduler = make_scheduler(optimizer, Iteration, gamma=GAMMA)
                phase = OPTIMIZER

            elif OPTIMIZER in ["adamw", "signed_adamw"]:
                optimizer = torch.optim.AdamW([dummy_data], lr=lr, weight_decay=1e-5)
                scheduler = make_scheduler(optimizer, Iteration, gamma=GAMMA)
                phase = OPTIMIZER

            else:
                raise ValueError(f"Unknown optimizer: {OPTIMIZER}")

            losses = []
            mses = []

            early_stop_reason = "fixed_iterations"
            early_stop_iter = Iteration
            best_loss_value = float("inf")
            best_dummy = None
            best_mse_value = None

            for iters in range(Iteration):
                if phase == "lbfgs":
                    def closure():
                        optimizer.zero_grad()
                        pred = net(dummy_data)
                        dummy_loss = criterion(pred, label_pred)
                        dummy_dy_dx = torch.autograd.grad(dummy_loss, selected_params, create_graph=True)
                        grad_diff, _ = compute_grad_match_loss(dummy_dy_dx, selected_original, selected_entry_masks=selected_entry_masks, grad_loss=GRAD_LOSS)
                        tv_loss = total_variation(dummy_data)
                        total_loss = grad_diff + TV_WEIGHT * tv_loss
                        total_loss.backward()
                        return total_loss

                    optimizer.step(closure)
                    current_loss = closure().item()

                    # Sigmoid-activation nets (LeNet, LeNet_bigger) are trained on raw [0,1]
                    # inputs with no normalisation. Clamping after each LBFGS step corrupts
                    # the quasi-Newton Hessian approximation (gradient stored at unclamped
                    # position, but next step starts from clamped position), causing many runs
                    # to diverge into wrong local minima. Skip clamping to match the original
                    # iDLG paper behaviour; pixels naturally converge to [0,1] when the loss
                    # drives them toward the GT.
                    if NETWORK_NAME not in {"LeNet", "LeNet_bigger"}:
                        with torch.no_grad():
                            dummy_data.clamp_(lower_bound, upper_bound)

                elif phase in ["adam", "adamw", "signed_adam", "signed_adamw"]:
                    optimizer.zero_grad()
                    pred = net(dummy_data)
                    dummy_loss = criterion(pred, label_pred)
                    dummy_dy_dx = torch.autograd.grad(dummy_loss, selected_params, create_graph=True)
                    grad_diff, num_terms = compute_grad_match_loss(dummy_dy_dx, selected_original, selected_entry_masks=selected_entry_masks, grad_loss=GRAD_LOSS)
                    tv_loss = total_variation(dummy_data)
                    total_loss = grad_diff + TV_WEIGHT * tv_loss
                    total_loss.backward()

                    if phase in ["signed_adam", "signed_adamw"] and dummy_data.grad is not None:
                        with torch.no_grad():
                            dummy_data.grad.sign_()

                    optimizer.step()
                    with torch.no_grad():
                        dummy_data.clamp_(lower_bound, upper_bound)

                    if scheduler is not None:
                        scheduler.step()

                    current_loss = total_loss.item()

                else:
                    raise ValueError(f"Unknown phase: {phase}")

                current_x = (dummy_data.detach() * ds + dm).clamp(0.0, 1.0)
                current_mse = torch.mean((current_x - gt_data) ** 2).item()

                if SAVE_GIF and iters % FRAME_INTERVAL == 0:
                    _restart_frames.append({
                        'iter': iters,
                        'dummy': current_x.cpu().numpy(),
                        'loss': current_loss if np.isfinite(current_loss) else float('inf'),
                        'mse': current_mse,
                    })
                    _last_gif_iter = iters

                if np.isfinite(current_loss) and current_loss < best_loss_value:
                    best_loss_value = current_loss
                    best_mse_value = current_mse
                    best_dummy = current_x.detach().clone()

                losses.append(current_loss)
                mses.append(current_mse)

                if current_loss < 1e-6:
                    early_stop_reason = "converged"
                    early_stop_iter = iters
                    break

                if iters % 1000 == 0:
                    current_lr = optimizer.param_groups[0]["lr"]
                    print(f'[GPU {device_id}] {OPTIMIZER} restart {restart_idx+1} iters {iters}, lr = {current_lr:.6g}, loss = {current_loss:.8f}, mse = {current_mse:.8f}')

            if SAVE_GIF and Iteration > 0 and _last_gif_iter != iters:
                _final_x = (dummy_data.detach() * ds + dm).clamp(0.0, 1.0)
                _restart_frames.append({
                    'iter': iters,
                    'dummy': _final_x.cpu().numpy(),
                    'loss': losses[-1] if losses else float('inf'),
                    'mse': torch.mean((_final_x - gt_data) ** 2).item(),
                })

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

            _best_mse_per_restart.append(best_restart_mse_value)
            _best_img_per_restart.append(best_restart_dummy.cpu().numpy() if best_restart_dummy is not None else None)
            _ssim_per_restart_list.append(
                compute_ssim_batch(best_restart_dummy, gt_data) if best_restart_dummy is not None else None
            )

        _psnr_per_restart[method] = [
            compute_psnr_from_mse(m, max_val=1.0) if (m is not None and np.isfinite(m)) else None
            for m in _best_mse_per_restart
        ]
        _img_per_restart[method] = _best_img_per_restart
        _mse_per_restart[method] = _best_mse_per_restart[:]
        _ssim_per_restart[method] = _ssim_per_restart_list

        if SAVE_GIF:
            init_frames_by_method[method] = _best_restart_init_np
            recon_frames_by_method[method] = _best_restart_frames

        if best_restart_dummy is not None:
            final_recon[method] = best_restart_dummy
            _best_ssim[method] = compute_ssim_batch(best_restart_dummy, gt_data)
        else:
            final_recon[method] = torch.zeros_like(gt_data)
            _best_ssim[method] = None

        _losses[method] = best_restart_losses if best_restart_losses is not None else [float("inf")]
        _labels[method] = label_pred.item()
        _mses[method] = best_restart_mses if best_restart_mses is not None else [float("inf")]
        _best_loss[method] = best_restart_loss_value
        _best_mse[method] = best_restart_mse_value
        _jac_rank[method] = jacobian_rank
        _jac_shape[method] = jacobian_shape

        early_stop_reason_dict[method] = best_restart_early_stop_reason
        early_stop_iter_dict[method] = best_restart_early_stop_iter

    result = {
        'idx_net': idx_net,
        'device_id': device_id,
        'gt_data': gt_data.detach().cpu().numpy(),
        'final_recon': {k: v.detach().cpu().numpy() for k, v in final_recon.items()},

        'last_psnr_idlg':    compute_psnr_from_mse(_mses['iDLG'][-1],        max_val=1.0) if 'iDLG'        in final_recon else None,
        'last_psnr_masked':  compute_psnr_from_mse(_mses['iDLG_masked'][-1], max_val=1.0) if 'iDLG_masked' in final_recon else None,
        'last_loss_iDLG':         _losses['iDLG'][-1]        if 'iDLG'        in final_recon else None,
        'last_mse_iDLG':          _mses['iDLG'][-1]          if 'iDLG'        in final_recon else None,
        'last_loss_iDLG_masked':  _losses['iDLG_masked'][-1] if 'iDLG_masked' in final_recon else None,
        'last_mse_iDLG_masked':   _mses['iDLG_masked'][-1]   if 'iDLG_masked' in final_recon else None,

        'best_psnr_idlg':   compute_psnr_from_mse(_best_mse.get('iDLG'),        max_val=1.0) if _best_mse.get('iDLG')        is not None else None,
        'best_psnr_masked': compute_psnr_from_mse(_best_mse.get('iDLG_masked'), max_val=1.0) if _best_mse.get('iDLG_masked') is not None else None,
        'best_loss_iDLG':        _best_loss.get('iDLG'),
        'best_mse_iDLG':         _best_mse.get('iDLG'),
        'best_loss_iDLG_masked': _best_loss.get('iDLG_masked'),
        'best_mse_iDLG_masked':  _best_mse.get('iDLG_masked'),
        'best_ssim_idlg':        _best_ssim.get('iDLG'),
        'best_ssim_masked':      _best_ssim.get('iDLG_masked'),

        'label_iDLG':        _labels.get('iDLG'),
        'label_iDLG_masked': _labels.get('iDLG_masked'),
        'jac_rank_iDLG':        _jac_rank.get('iDLG'),
        'jac_shape_iDLG':       _jac_shape.get('iDLG'),
        'jac_rank_iDLG_masked': _jac_rank.get('iDLG_masked'),
        'jac_shape_iDLG_masked':_jac_shape.get('iDLG_masked'),
        'gt_label': gt_label.detach().cpu().numpy(),
        'imidx_list': imidx_list,
        'early_stop_reason': early_stop_reason_dict,
        'early_stop_iter': early_stop_iter_dict,
        'psnr_per_restart_idlg':   _psnr_per_restart.get('iDLG'),
        'psnr_per_restart_masked': _psnr_per_restart.get('iDLG_masked'),
        'img_per_restart_idlg':    _img_per_restart.get('iDLG'),
        'img_per_restart_masked':  _img_per_restart.get('iDLG_masked'),
        'mse_per_restart_idlg':    _mse_per_restart.get('iDLG'),
        'mse_per_restart_masked':  _mse_per_restart.get('iDLG_masked'),
        'ssim_per_restart_idlg':   _ssim_per_restart.get('iDLG'),
        'ssim_per_restart_masked': _ssim_per_restart.get('iDLG_masked'),
        'restart_idx': SINGLE_RESTART_IDX,
        'init_frames': init_frames_by_method,
        'recon_frames': recon_frames_by_method,
    }

    result_queue.put(result)
