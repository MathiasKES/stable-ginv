# iDLG_mask.py
import os
import sys
import numpy as np
import torch
from torchvision import transforms
from datetime import datetime
import csv
import json

import torch.multiprocessing as mp
import argparse
from helper.visualization import save_recon_panel, save_recon_gif
from functions.io_utils import (paired_summary, baseline_key_from_args, load_baseline_registry,
    save_baseline_registry, update_idlg_baseline, write_baseline_summary_csv,
    parse_prefixes_with_fracs, masked_key_from_args, load_masked_registry,
    save_masked_registry, update_masked_registry)
from functions.Dataset import load_dataset
from run_single_exp import run_single_experiment
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mask_mode", type=str, default="gradsize_topfrac", help=(
        "Determines which gradient parameters are used during the masked iDLG reconstruction. "
        "Controls how the gradient mask is constructed before inverting gradients. "
        "Options:\n"
        "  'gradsize_topk'       - Keep only the top-K tensors ranked by gradient magnitude. "
                                "The exact number K is set via --gradsize_topk.\n"
        "  'gradsize_topfrac'    - Keep the top fraction of tensors by gradient magnitude. "
                                "The fraction is set via --gradsize_topfrac (e.g. 0.5 = top 50%%).\n"
        "  'gradsize_topk_entries'    - Keep the highest-ranked tensors until at least --gradsize_topk scalar gradient entries are retained."
        "  'gradsize_topfrac_entries' - Keep the highest-ranked tensors until at least the fraction --gradsize_topfrac of all scalar gradient entries are retained.\n"
        "  'gradsize_topfrac_entries_layer' - For EACH layer independently, keep its top --gradsize_topfrac fraction of scalar gradient entries. "
                                "Unlike gradsize_topfrac_entries (which selects globally and is dominated by large layers), "
                                "this mode preserves the same fraction from every layer, maintaining the relative distribution across layers.\n"
        "  'gradsize_topk_entries_layer' - For EACH layer independently, keep its top --gradsize_topk scalar gradient entries.\n"
        "  'prefix'              - Keep all parameters whose layer name starts with one of the "
                                "prefixes listed in --prefixes (e.g. 'conv1,linear').\n"
        "  'prefix_topk'           - First restrict to --prefixes, then keep the top-K tensors by gradient magnitude within those layers.\n"
        "  'prefix_topfrac'        - First restrict to --prefixes, then keep the top fraction of tensors by gradient magnitude within those layers.\n"
        "  'prefix_topk_entries'   - First restrict to --prefixes, then keep exactly the top-K scalar gradient entries globally.\n"
        "  'prefix_topfrac_entries'- First restrict to --prefixes, then keep exactly the top fraction of scalar gradient entries globally.\n"
        "  'prefix_topk_entries_layer'   - First restrict to --prefixes, then keep exactly the top-K scalar gradient entries for each layer.\n"
        "  'prefix_topfrac_entries_layer'- First restrict to --prefixes, then keep exactly the top fraction of scalar gradient entries for each layer.\n"
        "In all gradient-size modes, the metric used to measure gradient magnitude is controlled "
        "by --gradsize_metric."
    )) 

    parser.add_argument("--prefixes", type=str, default="conv1:1.0,layer1:1.0,layer2:1.0,layer3:1.0,fc:1.0", help=(
    "Comma-separated list of layer-name prefixes. "
    "For prefix_topfrac_entries_layer, each prefix can optionally include its own fraction. "
    "Format: 'conv1:1.0,layer1:0.5,layer2:0.3,layer3:0.2,fc:1.0'. "
    "For other prefix-based modes, only the prefix names are used. "
    "Examples: 'conv1,layer1,fc' or 'conv1:1.0,layer1:0.5,fc:1.0'."
    ))
    
    parser.add_argument("--gradsize_topk", type=int, default=20, help=(
        "Number of parameters (layers or individual tensors, depending on granularity) to retain "
        "when --mask_mode is 'gradsize_topk'. Parameters are ranked by their gradient magnitude "
        "as measured by --gradsize_metric, and only the K largest are kept in the mask; the rest "
        "are zeroed out. Larger values expose more of the gradient to the attacker, generally "
        "improving reconstruction quality. Has no effect when any other mask_mode is selected."
    ))

    parser.add_argument("--gradsize_topfrac", type=float, default=0.5, help=(
        "Fraction (between 0.0 and 1.0) of parameters to retain when --mask_mode is "
        "'gradsize_topfrac'. Parameters are ranked by gradient magnitude as measured by "
        "--gradsize_metric, and only the top fraction are kept; the rest are zeroed out. "
        "For example, 0.5 keeps the 50%% of parameters with the largest gradient norms, "
        "while 0.1 keeps only the top 10%%, making the attack more restricted. "
        "Has no effect when any other mask_mode is selected."
    ))

    parser.add_argument("--gradsize_metric", type=str, default="l2", help=(
        "The metric used to compute the scalar 'size' (magnitude) of each parameter's gradient "
        "tensor when ranking parameters in any 'gradsize_*' mask mode. Determines how a "
        "multi-dimensional gradient tensor (e.g. a conv weight of shape [C_out, C_in, kH, kW]) "
        "is reduced to a single comparable score. Options:\n"
        "  'l2'        - Frobenius / L2 norm of the gradient tensor (sqrt of sum of squares). "
                         "Sensitive to large individual values.\n"
        "  'mean_abs'  - Mean of the absolute values of all elements. Robust to outliers and "
                         "scales with the average gradient magnitude.\n"
        "  'sum_abs'   - Sum of absolute values (L1 norm). Similar to mean_abs but also grows "
                         "with the number of parameters in the tensor, favouring larger layers.\n"
        "Has no effect when mask_mode is 'prefix' or 'conv12_fc'."
    ))

    parser.add_argument("--lr", type=float, default=1, help=(
        "Learning rate for the gradient-inversion optimiser. Controls the step size used when "
        "updating the dummy data during each iteration of the iDLG (and masked iDLG) "
        "reconstruction loop. A higher learning rate can converge faster but may overshoot and "
        "produce unstable or noisy reconstructions. A lower learning rate gives smoother "
        "convergence at the cost of requiring more iterations. The default of 1.0 follows the "
        "original iDLG paper setting; tune together with --iteration when experimenting."
    ))

    parser.add_argument("--grad_loss", type=str, default="cos", choices=["cos", "l2"], help=(
    "Gradient matching loss: cos (cosine similarity) or l2 (squared distance). "
    "cos is default"
    ))

    parser.add_argument("--num_dummy", type=int, default=1, help=(
        "Number of dummy data samples to optimise simultaneously during each gradient-inversion "
        "attempt. This should match the batch size that was used when the victim computed the "
        "gradient being attacked. Setting this to 1 corresponds to the single-sample iDLG "
        "setting. Increasing it simulates attacking a larger batch, which is a harder problem "
        "and typically yields lower reconstruction quality per sample."
    ))

    parser.add_argument("--iteration", type=int, default=1000, help=(
        "Maximum number of optimisation iterations to run for each gradient-inversion attack "
        "(both the baseline iDLG and the masked iDLG variant). At each iteration the dummy "
        "image is updated to minimise the distance between its gradient and the observed "
        "gradient. Higher values allow more thorough optimisation at the cost of longer runtime."
    ))

    parser.add_argument("--num_exp", type=int, default=10, help=(
        "Total number of independent gradient-inversion experiments to run. Each experiment "
        "samples a different image from the dataset, computes its gradient on a freshly "
        "initialised network, and then attempts to reconstruct the original image via both "
        "the baseline iDLG attack and the masked iDLG variant. Aggregate statistics (average "
        "and median PSNR, loss, MSE) are computed across all experiments and written to the "
        "CSV results file. Experiments are distributed across available GPUs in parallel."
    ))

    parser.add_argument("--network", type=str, default="resnet18", help=(
        "Name of the neural-network architecture whose gradients will be attacked. The chosen "
        "network is instantiated with random weights, a single forward/backward pass is "
        "performed on a ground-truth sample to obtain the gradient, and that gradient is then "
        "fed to the iDLG inversion attack. Supported options include: 'LeNet', 'LeNet_bigger', "
        "'MediumCNN', and 'resnetxx'. The architecture must be compatible with the image shape "
        "and number of classes implied by --dataset. The architecture also affects which "
        "mask modes are meaningful."
    ))

    parser.add_argument("--dataset", type=str, default="cifar100", help=(
        "Dataset to draw ground-truth images from. Determines image shape, number of channels, "
        "and number of classes used throughout the experiment. Supported options:\n"
        "  'MNIST'    - 28x28 greyscale, 10 classes.\n"
        "  'cifar10'  - 32x32 RGB, 10 classes.\n"
        "  'cifar100' - 32x32 RGB, 100 classes.\n"
        "  'lfw'      - Labelled Faces in the Wild, resized to 32x32 RGB, 5749 identity classes.\n"
        "The dataset is downloaded automatically to ./data/ (or the HPC path) if not already present."
    ))

    parser.add_argument("--run_id", type=int, default=0, help=(
        "Integer identifier for this particular run. Used as a seed offset or index when "
        "selecting which images from the dataset are attacked across experiments, ensuring "
        "that different run_ids sample non-overlapping (or differently-ordered) subsets of "
        "the dataset. Useful for running multiple independent batches of experiments without "
        "repeating the same images, and for reproducibility when comparing results across "
        "configurations."
    ))

    parser.add_argument("--methods", type=str, default="idlg", choices=["idlg", "masked", "both"], help=(
    "Which reconstruction method(s) to run.\n"
    "  'idlg'   - run only the baseline iDLG attack.\n"
    "  'masked' - run only the masked iDLG attack.\n"
    "  'both'   - run both baseline and masked iDLG."
))
    
    parser.add_argument("--compute_jacobian_rank", action="store_true", help=(
    "If set, compute the Jacobian of the observed gradient vector with respect to the input "
    "and report its rank. This can be very expensive, especially for ResNet18."
))
    
    parser.add_argument("--jacobian_max_entries", type=int, default=4000, help=(
    "Maximum number of observed gradient entries to use when computing the Jacobian rank."
))
    
    parser.add_argument("--jacobian_select_mode", type=str, default="topk_abs",
                    choices=["topk_abs", "first", "random"], help=(
    "How to choose which observed gradient entries are used for Jacobian rank computation."
))
    
    parser.add_argument("--tv_weight", type=float, default=0.0, help=(
    "Weight of total variation regularization added to the reconstruction loss. "
    "A small positive value can reduce noise and encourage smoother images."
))
    
    parser.add_argument("--optimizer", type=str, default="lbfgs", choices=["lbfgs", "adam", "adamw", "signed_adam", "signed_adamw"], help="Optimizer used for the reconstruction of dummy_data."
)
    
    parser.add_argument("--num_restarts", type=int, default=1,
        help="Number of random restarts for each reconstruction.")
    
    parser.add_argument(
        "--max_iteration",
        type=int,
        default=20,
        help="Maximum number of iterations per LBFGS step. Only used when --optimizer lbfgs."
    )

    parser.add_argument(
        "--history_size",
        type=int,
        default=100,
        help="History size for LBFGS. Only used when --optimizer lbfgs."
    )

    parser.add_argument("--save_gif", action="store_true",
        help="Save an animated GIF showing reconstruction progress per experiment.")

    
    parser.add_argument("--pretrained", action="store_true", 
        help="Load ImageNet-pretrained torchvision weights for supported networks.")
    
    parser.add_argument("--gamma", type=float, default=0.5, help="gamma for learning rate scheduler")

    args = parser.parse_args()

    if args.optimizer != "lbfgs":
        if "--max_iteration" in sys.argv or "--history_size" in sys.argv:
            parser.error("--max_iteration and --history_size can only be used when --optimizer lbfgs")

    if not (0.0 < args.gradsize_topfrac <= 1.0):
        parser.error(f"--gradsize_topfrac must be in (0, 1], got {args.gradsize_topfrac}")

    # -------- Masking config --------
    MASK_MODE = args.mask_mode
    PREFIXES, PREFIX_LAYER_FRACS = parse_prefixes_with_fracs(args.prefixes)
    GRADSIZE_TOPK = args.gradsize_topk
    GRADSIZE_TOPFRAC = args.gradsize_topfrac
    GRADSIZE_METRIC = args.gradsize_metric
    GRAD_LOSS = args.grad_loss
    METHODS = args.methods
    SAVE_GIF = args.save_gif
    Iteration = args.iteration
    FRAME_INTERVAL = max(1, (Iteration + 79) // 80)  # target ≤80 frames → ≤10 s at 8 fps
    GIF_FPS = 8
    COMPUTE_JACOBIAN_RANK = args.compute_jacobian_rank
    JACOBIAN_MAX_ENTRIES = args.jacobian_max_entries
    JACOBIAN_SELECT_MODE = args.jacobian_select_mode
    TV_WEIGHT = args.tv_weight
    OPTIMIZER = args.optimizer
    GAMMA = args.gamma
    lr = args.lr
    num_dummy = args.num_dummy
    num_exp = args.num_exp
    NETWORK_NAME = args.network
    NUM_RESTARTS = args.num_restarts
    MAX_ITERATION = args.max_iteration
    HISTORY_SIZE = args.history_size
    NETWORK_TRAINED = args.pretrained
    dataset = args.dataset
    run_id = args.run_id
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

    try:
        os.makedirs(data_path, mode=0o770, exist_ok=True)
        os.makedirs(save_path, mode=0o770, exist_ok=True)
    except Exception as e:
        print(f"Warning: failed to set permissions for directories: {e}")
    
    baseline_dir = os.path.join(save_path, "baselines")
    baseline_registry_path = os.path.join(baseline_dir, "idlg_baselines_registry.json")
    baseline_summary_csv_path = os.path.join(baseline_dir, "idlg_baselines_summary.csv")
    masked_registry_path = os.path.join(baseline_dir, "masked_registry.json")

    print(dataset, 'root_path:', root_path)
    print(dataset, 'data_path:', data_path)
    print(dataset, 'save_path:', save_path)

    # -------- load data --------
    dst, channel, num_classes, shape_img = load_dataset(dataset, data_path)


    # -------- panel buffers --------
    panel_block_size = num_exp
    panel_block_idx = 0
    panel_gt_pil = []
    panel_idlg_pil = []
    panel_masked_pil = []
    panel_png_paths = []
    panel_psnr_idlg = []
    panel_ssim_idlg = []
    panel_psnr_masked = []
    panel_ssim_masked = []

    psnr_idlg_all = []
    psnr_masked_all = []
    final_loss_idlg_all = []
    final_mse_idlg_all = []
    final_loss_masked_all = []
    final_mse_masked_all = []

    best_psnr_idlg_all = []
    best_psnr_masked_all = []
    best_loss_idlg_all = []
    best_mse_idlg_all = []
    best_loss_masked_all = []
    best_mse_masked_all = []
    best_ssim_idlg_all = []
    best_ssim_masked_all = []

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
        'run_id': run_id,
        'MASK_MODE': MASK_MODE,
        'PREFIXES': PREFIXES,
        'PREFIX_LAYER_FRACS': PREFIX_LAYER_FRACS,
        'GRADSIZE_TOPK': GRADSIZE_TOPK,
        'GRADSIZE_TOPFRAC': GRADSIZE_TOPFRAC,
        'GRADSIZE_METRIC': GRADSIZE_METRIC,
        'GRAD_LOSS': GRAD_LOSS,
        "GAMMA": GAMMA,
        'NETWORK_NAME': NETWORK_NAME,
'METHODS': METHODS,
        'COMPUTE_JACOBIAN_RANK': COMPUTE_JACOBIAN_RANK,
        'JACOBIAN_MAX_ENTRIES': JACOBIAN_MAX_ENTRIES,
        'JACOBIAN_SELECT_MODE': JACOBIAN_SELECT_MODE,
        'TV_WEIGHT': TV_WEIGHT,
        'OPTIMIZER': OPTIMIZER,
        "NUM_RESTARTS": NUM_RESTARTS,
        'MAX_ITERATION': MAX_ITERATION,
        'HISTORY_SIZE': HISTORY_SIZE,
        'NETWORK_TRAINED': NETWORK_TRAINED,
        'SAVE_GIF': SAVE_GIF,
        'FRAME_INTERVAL': FRAME_INTERVAL,
    }

    # -------- Run experiments in parallel --------
    unknowns = channel * shape_img[0] * shape_img[1]
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
    all_results_by_idx = {}

    # Start one experiment per GPU initially
    for device_id in range(min(num_gpus, num_exp)):
        p = mp.Process(
            target=run_single_experiment,
            args=(next_exp, device_id, dst, dataset, config, result_queue)
        )
        p.start()
        active_processes[device_id] = p
        next_exp += 1

    # Keep launching a new experiment whenever one finishes
    with tqdm(total=num_exp, desc="Experiments", position=0) as pbar:
        while completed < num_exp:
            result = result_queue.get()
            completed += 1
            pbar.update(1)
            pbar.set_postfix(last_exp=result["idx_net"], gpu=result["device_id"])

            idx_net = result['idx_net']
            finished_device = result['device_id']

            if result.get('error') is not None:
                print(f"\n[ERROR] Experiment {idx_net} on GPU {finished_device} failed:")
                print(result['traceback'])
                active_processes[finished_device].join()
                if next_exp < num_exp:
                    p = mp.Process(
                        target=run_single_experiment,
                        args=(next_exp, finished_device, dst, dataset, config, result_queue)
                    )
                    p.start()
                    active_processes[finished_device] = p
                    tqdm.write(f"Launching experiment {next_exp} on GPU {finished_device}")
                    next_exp += 1
                continue

            all_results_by_idx[idx_net] = result

            if result.get('best_ssim_idlg') is not None:
                best_ssim_idlg_all.append(result['best_ssim_idlg'])
            if result.get('best_ssim_masked') is not None:
                best_ssim_masked_all.append(result['best_ssim_masked'])
            if result.get('last_psnr_idlg') is not None:
                psnr_idlg_all.append(result['last_psnr_idlg'])
            if result.get('last_psnr_masked') is not None:
                psnr_masked_all.append(result['last_psnr_masked'])

            if result.get('last_loss_iDLG') is not None:
                final_loss_idlg_all.append(result['last_loss_iDLG'])
            if result.get('last_mse_iDLG') is not None:
                final_mse_idlg_all.append(result['last_mse_iDLG'])
            if result.get('last_loss_iDLG_masked') is not None:
                final_loss_masked_all.append(result['last_loss_iDLG_masked'])
            if result.get('last_mse_iDLG_masked') is not None:
                final_mse_masked_all.append(result['last_mse_iDLG_masked'])

            if result.get('best_psnr_idlg') is not None:
                best_psnr_idlg_all.append(result['best_psnr_idlg'])
            if result.get('best_psnr_masked') is not None:
                best_psnr_masked_all.append(result['best_psnr_masked'])

            if result.get('best_loss_iDLG') is not None:
                best_loss_idlg_all.append(result['best_loss_iDLG'])
            if result.get('best_mse_iDLG') is not None:
                best_mse_idlg_all.append(result['best_mse_iDLG'])
            if result.get('best_loss_iDLG_masked') is not None:
                best_loss_masked_all.append(result['best_loss_iDLG_masked'])
            if result.get('best_mse_iDLG_masked') is not None:
                best_mse_masked_all.append(result['best_mse_iDLG_masked'])

            # ---- accumulate recon panel ----
            gt_pil = tp(torch.from_numpy(result['gt_data'])[0])
            panel_gt_pil.append(gt_pil)

            if 'iDLG' not in result['final_recon']:
                idlg_pil = gt_pil
            elif result.get('best_psnr_idlg') is None:
                tqdm.write(f"[WARNING] Experiment {idx_net}: iDLG reconstruction failed (all restarts diverged); showing blank in panel.")
                idlg_pil = gt_pil
            else:
                idlg_pil = tp(torch.from_numpy(result['final_recon']['iDLG'])[0])
            panel_idlg_pil.append(idlg_pil)

            if 'iDLG_masked' not in result['final_recon']:
                masked_pil = gt_pil
            elif result.get('best_psnr_masked') is None:
                tqdm.write(f"[WARNING] Experiment {idx_net}: masked iDLG reconstruction failed (all restarts diverged); showing blank in panel.")
                masked_pil = gt_pil
            else:
                masked_pil = tp(torch.from_numpy(result['final_recon']['iDLG_masked'])[0])
            panel_masked_pil.append(masked_pil)

            panel_psnr_idlg.append(result.get('best_psnr_idlg'))
            panel_ssim_idlg.append(result.get('best_ssim_idlg'))
            panel_psnr_masked.append(result.get('best_psnr_masked'))
            panel_ssim_masked.append(result.get('best_ssim_masked'))

            if len(panel_gt_pil) == panel_block_size:
                panel_path = save_recon_panel(
                params, panel_gt_pil, panel_idlg_pil, panel_masked_pil,
                save_path, panel_block_idx, dataset, mask_desc, timestamp_str,
                methods=METHODS,
                psnr_idlg=panel_psnr_idlg, ssim_idlg=panel_ssim_idlg,
                psnr_masked=panel_psnr_masked, ssim_masked=panel_ssim_masked,
            )
                if panel_path:
                    panel_png_paths.append(panel_path)
                panel_block_idx += 1
                panel_gt_pil.clear()
                panel_idlg_pil.clear()
                panel_masked_pil.clear()
                panel_psnr_idlg.clear()
                panel_ssim_idlg.clear()
                panel_psnr_masked.clear()
                panel_ssim_masked.clear()

            es_r = result.get("early_stop_reason", {})
            es_i = result.get("early_stop_iter", {})

            print(f"early_stop iDLG: {es_r.get('iDLG')} @ {es_i.get('iDLG')}")

            if result.get('last_loss_iDLG') is not None:
                print('last_loss_iDLG:', result['last_loss_iDLG'], 'last_mse_iDLG:', result['last_mse_iDLG'])
            if result.get('best_loss_iDLG') is not None:
                print('best_loss_iDLG:', result['best_loss_iDLG'], 'best_mse_iDLG:', result['best_mse_iDLG'])

            if result.get('last_loss_iDLG_masked') is not None:
                print('last_loss_iDLG_masked:', result['last_loss_iDLG_masked'], 'last_mse_iDLG_masked:', result['last_mse_iDLG_masked'])
            if result.get('best_loss_iDLG_masked') is not None:
                print('best_loss_iDLG_masked:', result['best_loss_iDLG_masked'], 'best_mse_iDLG_masked:', result['best_mse_iDLG_masked'])
                
            if result.get('jac_rank_iDLG') is not None:
                print('jac_rank_iDLG:', result['jac_rank_iDLG'],
                    'jac_shape_iDLG:', result['jac_shape_iDLG'])

            if result.get('jac_rank_iDLG_masked') is not None:
                print('jac_rank_iDLG_masked:', result['jac_rank_iDLG_masked'],
                    'jac_shape_iDLG_masked:', result['jac_shape_iDLG_masked'])

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
                tqdm.write(f"Launching experiment {next_exp} on GPU {finished_device}")
                next_exp += 1

    # Final cleanup
    for p in active_processes.values():
        p.join()
    # save any remaining
    if len(panel_gt_pil) > 0:
        panel_path = save_recon_panel(
        params, panel_gt_pil, panel_idlg_pil, panel_masked_pil,
        save_path, panel_block_idx, dataset, mask_desc, timestamp_str,
        methods=METHODS,
        psnr_idlg=panel_psnr_idlg, ssim_idlg=panel_ssim_idlg,
        psnr_masked=panel_psnr_masked, ssim_masked=panel_ssim_masked,
    )
        if panel_path:
            panel_png_paths.append(panel_path)

    # -------- Save animated GIFs --------
    if SAVE_GIF:
        ordered_results = [all_results_by_idx[i] for i in sorted(all_results_by_idx.keys())]
        for b_start in range(0, len(ordered_results), panel_block_size):
            block = ordered_results[b_start:b_start + panel_block_size]
            b_idx = b_start // panel_block_size
            save_recon_gif(
                block, save_path, b_idx, dataset, mask_desc, timestamp_str,
                methods=METHODS, fps=GIF_FPS,
            )

    # -------- Paired statistics --------
    empty_stats = {
        "n": float("nan"),
        "mean_diff": float("nan"),
        "std_diff": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "t_stat": float("nan"),
        "p_value": float("nan"),
    }

    mse_paired_stats = empty_stats.copy()
    psnr_paired_stats = empty_stats.copy()
    mse_ci_str = ""
    psnr_ci_str = ""
    mse_significant_str = ""
    psnr_significant_str = ""
    psnr_normality_str = ""
    mse_normality_str = ""

    baseline_key, comparable_args = baseline_key_from_args(args)

    if METHODS == "both":
        paired_best_mse_idlg = []
        paired_best_mse_masked = []
        paired_best_psnr_idlg = []
        paired_best_psnr_masked = []

        for idx in sorted(all_results_by_idx):
            result = all_results_by_idx[idx]

            mse_idlg = result.get("best_mse_iDLG")
            mse_masked = result.get("best_mse_iDLG_masked")
            psnr_idlg = result.get("best_psnr_idlg")
            psnr_masked = result.get("best_psnr_masked")

            if mse_idlg is not None and mse_masked is not None and np.isfinite(mse_idlg) and np.isfinite(mse_masked):
                paired_best_mse_idlg.append(mse_idlg)
                paired_best_mse_masked.append(mse_masked)

            if psnr_idlg is not None and psnr_masked is not None and np.isfinite(psnr_idlg) and np.isfinite(psnr_masked):
                paired_best_psnr_idlg.append(psnr_idlg)
                paired_best_psnr_masked.append(psnr_masked)

        mse_summary = paired_summary(
            np.array(paired_best_mse_masked),
            np.array(paired_best_mse_idlg),
            metric="mse",
            confidence=0.95,
            ci_decimals=10,
        )

        psnr_summary = paired_summary(
            np.array(paired_best_psnr_masked),
            np.array(paired_best_psnr_idlg),
            metric="psnr",
            confidence=0.95,
            ci_decimals=5,
        )

        mse_paired_stats = mse_summary["stats"]
        psnr_paired_stats = psnr_summary["stats"]
        mse_ci_str = mse_summary["ci_str"]
        psnr_ci_str = psnr_summary["ci_str"]
        mse_significant_str = mse_summary["significant_str"]
        psnr_significant_str = psnr_summary["significant_str"]
        psnr_normality_str = psnr_summary["normality_str"]
        mse_normality_str = mse_summary["normality_str"]

    elif METHODS == "masked":
        registry = load_baseline_registry(baseline_registry_path)

        if baseline_key not in registry:
            print("\nWARNING: No matching iDLG baseline found for this masked run.")
            print("Run the same command with --methods idlg first, using the same non-mask arguments.")
            print(f"Expected baseline key: {baseline_key}")
        else:
            baseline_entry = registry[baseline_key]
            baseline_psnr_list = baseline_entry["best_psnr_list"]
            baseline_mse_list = baseline_entry["best_mse_list"]
            n_total = len(baseline_psnr_list)

            paired_best_psnr_idlg = []
            paired_best_psnr_masked = []
            paired_best_mse_idlg = []
            paired_best_mse_masked = []

            for idx in sorted(all_results_by_idx):
                result = all_results_by_idx[idx]

                psnr_baseline = baseline_psnr_list[idx]
                psnr_masked = result.get("best_psnr_masked")
                if psnr_masked is not None and np.isfinite(psnr_masked) and np.isfinite(psnr_baseline):
                    paired_best_psnr_idlg.append(psnr_baseline)
                    paired_best_psnr_masked.append(psnr_masked)

                mse_baseline = baseline_mse_list[idx]
                mse_masked = result.get("best_mse_iDLG_masked")
                if mse_masked is not None and np.isfinite(mse_masked) and np.isfinite(mse_baseline):
                    paired_best_mse_idlg.append(mse_baseline)
                    paired_best_mse_masked.append(mse_masked)

            n_psnr = len(paired_best_psnr_masked)
            n_mse = len(paired_best_mse_masked)
            if n_psnr < n_total:
                print(f"WARNING: {n_total - n_psnr}/{n_total} experiment(s) excluded from PSNR paired test (crashed or non-finite).")
            if n_mse < n_total:
                print(f"WARNING: {n_total - n_mse}/{n_total} experiment(s) excluded from MSE paired test (crashed or non-finite).")

            mse_summary = paired_summary(
                np.array(paired_best_mse_masked),
                np.array(paired_best_mse_idlg),
                metric="mse",
                confidence=0.95,
                ci_decimals=10,
            )

            psnr_summary = paired_summary(
                np.array(paired_best_psnr_masked),
                np.array(paired_best_psnr_idlg),
                metric="psnr",
                confidence=0.95,
                ci_decimals=5,
            )

            mse_paired_stats = mse_summary["stats"]
            psnr_paired_stats = psnr_summary["stats"]
            mse_ci_str = mse_summary["ci_str"]
            psnr_ci_str = psnr_summary["ci_str"]
            mse_significant_str = mse_summary["significant_str"]
            psnr_significant_str = psnr_summary["significant_str"]
            psnr_normality_str = psnr_summary["normality_str"]
            mse_normality_str = mse_summary["normality_str"]

            print(f"\nLoaded iDLG baseline (run_id={run_id}). Paired test uses {n_psnr}/{n_total} experiment(s).")

    # -------- Compute statistics --------
    avg_psnr_idlg           = float(np.mean(psnr_idlg_all))             if len(psnr_idlg_all)       else float("nan")
    avg_psnr_masked         = float(np.mean(psnr_masked_all))           if len(psnr_masked_all)     else float("nan")
    std_psnr_idlg           = float(np.std(psnr_idlg_all, ddof=1))      if len(psnr_idlg_all) > 1   else float("nan")
    std_psnr_masked         = float(np.std(psnr_masked_all, ddof=1))    if len(psnr_masked_all) > 1 else float("nan")

    avg_final_loss_idlg     = float(np.mean(final_loss_idlg_all))       if final_loss_idlg_all      else float("nan")
    avg_final_mse_idlg      = float(np.mean(final_mse_idlg_all))        if final_mse_idlg_all       else float("nan")
    avg_final_loss_masked   = float(np.mean(final_loss_masked_all))     if final_loss_masked_all    else float("nan")
    avg_final_mse_masked    = float(np.mean(final_mse_masked_all))      if final_mse_masked_all     else float("nan")

    med_final_loss_idlg     = float(np.median(final_loss_idlg_all))     if final_loss_idlg_all      else float("nan")
    med_final_mse_idlg      = float(np.median(final_mse_idlg_all))      if final_mse_idlg_all       else float("nan")
    med_final_loss_masked   = float(np.median(final_loss_masked_all))   if final_loss_masked_all    else float("nan")
    med_final_mse_masked    = float(np.median(final_mse_masked_all))    if final_mse_masked_all     else float("nan")

    avg_best_psnr_idlg         = float(np.mean(best_psnr_idlg_all))         if best_psnr_idlg_all else float("nan")
    avg_best_psnr_masked       = float(np.mean(best_psnr_masked_all))       if best_psnr_masked_all else float("nan")
    std_best_psnr_idlg = float(np.std(best_psnr_idlg_all, ddof=1)) if len(best_psnr_idlg_all) > 1 else float("nan")
    std_best_psnr_masked = float(np.std(best_psnr_masked_all, ddof=1)) if len(best_psnr_masked_all) > 1 else float("nan")

    avg_best_loss_idlg         = float(np.mean(best_loss_idlg_all))         if best_loss_idlg_all else float("nan")
    avg_best_mse_idlg          = float(np.mean(best_mse_idlg_all))          if best_mse_idlg_all else float("nan")
    avg_best_loss_masked       = float(np.mean(best_loss_masked_all))       if best_loss_masked_all else float("nan")
    avg_best_mse_masked        = float(np.mean(best_mse_masked_all))        if best_mse_masked_all else float("nan")

    med_best_loss_idlg         = float(np.median(best_loss_idlg_all))       if best_loss_idlg_all else float("nan")
    med_best_mse_idlg          = float(np.median(best_mse_idlg_all))        if best_mse_idlg_all else float("nan")
    med_best_loss_masked       = float(np.median(best_loss_masked_all))     if best_loss_masked_all else float("nan")
    med_best_mse_masked        = float(np.median(best_mse_masked_all))      if best_mse_masked_all else float("nan")


    avg_best_ssim_idlg   = float(np.mean(best_ssim_idlg_all))   if best_ssim_idlg_all   else float("nan")
    avg_best_ssim_masked = float(np.mean(best_ssim_masked_all)) if best_ssim_masked_all else float("nan")
    std_best_ssim_idlg   = float(np.std(best_ssim_idlg_all, ddof=1))   if len(best_ssim_idlg_all) > 1   else float("nan")
    std_best_ssim_masked = float(np.std(best_ssim_masked_all, ddof=1)) if len(best_ssim_masked_all) > 1 else float("nan")
    # -------- Save/update iDLG baseline registry --------
    if METHODS in ("idlg", "both"):
        ordered_best_psnr_idlg = []
        ordered_best_mse_idlg = []

        for idx in sorted(all_results_by_idx):
            result = all_results_by_idx[idx]

            psnr = result.get("best_psnr_idlg")
            mse = result.get("best_mse_iDLG")

            if psnr is None or mse is None or not np.isfinite(psnr) or not np.isfinite(mse):
                raise ValueError(f"Missing or invalid iDLG baseline result for experiment idx={idx}")

            ordered_best_psnr_idlg.append(float(psnr))
            ordered_best_mse_idlg.append(float(mse))

        registry = load_baseline_registry(baseline_registry_path)

        updated_entry = update_idlg_baseline(
            registry,
            baseline_key,
            comparable_args,
            ordered_best_psnr_idlg,
            ordered_best_mse_idlg,
        )

        save_baseline_registry(baseline_registry_path, registry)
        write_baseline_summary_csv(baseline_summary_csv_path, registry)

        print(f"\nSaved iDLG baseline:")
        print(f"baseline_key: {baseline_key}")
        print(f"registry: {baseline_registry_path}")
        print(f"summary csv: {baseline_summary_csv_path}")

    if METHODS in ["masked", "both"] and best_psnr_masked_all:
        masked_registry = load_masked_registry(masked_registry_path)
        masked_key, masked_comparable_args = masked_key_from_args(args)
        update_masked_registry(
            masked_registry,
            masked_key,
            masked_comparable_args,
            best_psnr_masked_all,
            best_mse_masked_all,
        )
        save_masked_registry(masked_registry_path, masked_registry)
        print(f"\nSaved masked registry entry:")
        print(f"masked_key: {masked_key}")
        print(f"registry: {masked_registry_path}")

    csv_path = os.path.join(save_path, "exp_results.csv")
    file_exists = os.path.isfile(csv_path)

    png_path_str = "|".join(panel_png_paths)

    common = {
        "timestamp": timestamp_str,
        "job_id": "INTERACTIVE" if os.environ.get("LSB_INTERACTIVE") == "Y" else os.environ.get("LSB_JOBID", ""),
        "Run by": os.environ.get("USER", ""),
        "device": os.environ.get("LSB_QUEUE", ""),
        "cmd": "python " + " ".join(sys.argv),
        "dataset": dataset,
        "network": NETWORK_NAME,
        "pretrained": NETWORK_TRAINED,
        "restarts": NUM_RESTARTS,
        "lr": lr,
        "gamma": GAMMA,
        "iteration": Iteration,
        "num_exp": num_exp,
        "tv_weight": TV_WEIGHT,
        "optimizer": OPTIMIZER,
        "max_iter": MAX_ITERATION,
        "history": HISTORY_SIZE}
    
    grad_value = ""
    if MASK_MODE in [
        "gradsize_topfrac",
        "gradsize_topfrac_entries",
        "gradsize_topfrac_entries_layer",
        "prefix_topfrac",
        "prefix_topfrac_entries",
        "prefix_topfrac_entries_layer",
    ]:
        grad_value = GRADSIZE_TOPFRAC
    elif MASK_MODE in [
        "gradsize_topk",
        "gradsize_topk_entries",
        "gradsize_topk_entries_layer",
        "prefix_topk",
        "prefix_topk_entries",
        "prefix_topk_entries_layer",
    ]:
        grad_value = GRADSIZE_TOPK

    rows = []

    if METHODS in ["idlg", "both"]:
        rows.append({
            "method": "iDLG",
            **common,
            "mask_mode": "",
            "grad_param": "",
            "med_best_loss": round(med_best_loss_idlg,5),
            "avg_best_loss": round(avg_best_loss_idlg,5),
            "med_best_mse": round(med_best_mse_idlg,10),
            "avg_best_mse": round(avg_best_mse_idlg,10),
            "avg_best_psnr": round(avg_best_psnr_idlg,5),
            "std_best_psnr": round(std_best_psnr_idlg,5),
            "avg_best_ssim": round(avg_best_ssim_idlg,5),
            "std_best_ssim": round(std_best_ssim_idlg,5),
            "png_path": png_path_str,
        })

    if METHODS in ["masked", "both"]:
        rows.append({
            "method": "iDLG_masked",
            **common,
            "mask_mode": MASK_MODE,
            "prefixes": args.prefixes if "prefix" in MASK_MODE else "",
            "grad_param": grad_value,
            "mse_ci": mse_ci_str if METHODS in ["both", "masked"] else "",
            "mse_significant": mse_significant_str if METHODS in ["both", "masked"] else "",
            "psnr_ci": psnr_ci_str if METHODS in ["both", "masked"] else "",
            "psnr_significant": psnr_significant_str if METHODS in ["both", "masked"] else "",
            "psnr_normality": psnr_normality_str if METHODS in ["both", "masked"] else "",
            "mse_normality": mse_normality_str if METHODS in ["both", "masked"] else "",
            "med_best_loss": round(med_best_loss_masked,5),
            "avg_best_loss": round(avg_best_loss_masked,5),
            "med_best_mse": round(med_best_mse_masked,10),
            "avg_best_mse": round(avg_best_mse_masked,10),
            "avg_best_psnr": round(avg_best_psnr_masked,5),
            "std_best_psnr": round(std_best_psnr_masked,5),
            "avg_best_ssim": round(avg_best_ssim_masked,5),
            "std_best_ssim": round(std_best_ssim_masked,5),
            "png_path": png_path_str,
        })

    fieldnames = ["method"] + list(common.keys()) + [
        "mask_mode", "prefixes", "grad_param",
        "med_best_loss", "avg_best_loss", "med_best_mse", "avg_best_mse",
        "avg_best_psnr", "std_best_psnr",
        "avg_best_ssim", "std_best_ssim",
        "mse_ci", "mse_significant", "psnr_ci", "psnr_significant", "psnr_normality", "mse_normality",
        "png_path",
    ]

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
    print(f"Gamma for lr scheduler was: {GAMMA}")
    if MASK_MODE in ("gradsize_topfrac", "gradsize_topfrac_entries", "gradsize_topfrac_entries_layer"):
        print(f"top fraction kept ({MASK_MODE}): {GRADSIZE_TOPFRAC*100}%")
    elif MASK_MODE in ("gradsize_topk", "gradsize_topk_entries", "gradsize_topk_entries_layer"):
        print(f"top-k kept ({MASK_MODE}): {GRADSIZE_TOPK}")
    print(f"Avg final loss iDLG: {avg_final_loss_idlg:.6f} | masked: {avg_final_loss_masked:.6f}")
    print(f"Avg final mse  iDLG: {avg_final_mse_idlg:.8f} | masked: {avg_final_mse_masked:.8f}")
    print(f"Median final loss iDLG: {med_final_loss_idlg:.6f} | masked: {med_final_loss_masked:.6f}")
    print(f"Median final mse  iDLG: {med_final_mse_idlg:.8f} | masked: {med_final_mse_masked:.8f}")
    print(f"Average PSNR iDLG: {avg_psnr_idlg:.4f} ± {std_psnr_idlg:.4f} dB | masked: {avg_psnr_masked:.4f} ± {std_psnr_masked:.4f} dB")
    print(f"Avg best loss iDLG: {avg_best_loss_idlg:.6f} | masked: {avg_best_loss_masked:.6f}")
    print(f"Avg best mse  iDLG: {avg_best_mse_idlg:.8f} | masked: {avg_best_mse_masked:.8f}")
    print(f"Median best loss iDLG: {med_best_loss_idlg:.6f} | masked: {med_best_loss_masked:.6f}")
    print(f"Median best mse  iDLG: {med_best_mse_idlg:.10f} | masked: {med_best_mse_masked:.10f}")
    print(f"Average best PSNR iDLG: {avg_best_psnr_idlg:.4f} ± {std_best_psnr_idlg:.4f} dB | masked: {avg_best_psnr_masked:.4f} ± {std_best_psnr_masked:.4f} dB")
    print(f"Average best SSIM iDLG: {avg_best_ssim_idlg:.4f} ± {std_best_ssim_idlg:.4f} | masked: {avg_best_ssim_masked:.4f} ± {std_best_ssim_masked:.4f}")
    if METHODS in ["both", "masked"]:
        print(
            f"Paired best MSE diff (masked - iDLG): {mse_paired_stats['mean_diff']:.10f} "
            f"| 95% CI: {mse_ci_str} "
            f"| significant: {mse_significant_str} "
            f"| normality: {mse_normality_str}")
        print(
            f"Paired best PSNR diff (masked - iDLG): {psnr_paired_stats['mean_diff']:.5f} dB "
            f"| 95% CI: {psnr_ci_str} "
            f"| significant: {psnr_significant_str} "
            f"| normality: {psnr_normality_str}")

    print("Job resource usage:")
    print("Max memory allocated:", torch.cuda.memory.max_memory_allocated() / (1024 ** 3), "GB")
    print("Max memory reserved:", torch.cuda.memory.max_memory_reserved() / (1024 ** 3), "GB")

    print("Memory summary:")
    print(torch.cuda.memory.memory_summary())

        
if __name__ == '__main__':
    main()