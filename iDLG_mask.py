import os
import sys
import numpy as np
import torch
from torchvision import datasets, transforms
from datetime import datetime
import csv
import torch.multiprocessing as mp
import argparse
from Misc_functions import save_recon_panel
from Dataset import lfw_dataset
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
        "  'gradsize_threshold'  - Keep only tensors whose gradient magnitude exceeds a fixed "
                                "threshold, set via --gradsize_threshold.\n"
        "  'gradsize_topk_entries'    - Keep the highest-ranked tensors until at least --gradsize_topk scalar gradient entries are retained."
        "  'gradsize_topfrac_entries' - Keep the highest-ranked tensors until at least the fraction --gradsize_topfrac of all scalar gradient entries are retained."                        
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

    parser.add_argument("--gradsize_threshold", type=float, default=None, help=(
        "Absolute magnitude threshold for gradient masking when --mask_mode is "
        "'gradsize_threshold'. Any parameter whose gradient magnitude (as measured by "
        "--gradsize_metric) is strictly below this value is zeroed out; all parameters at or "
        "above the threshold are retained. If set to None (default), no threshold is applied. "
        "Unlike topk/topfrac modes, this threshold is data-independent and may retain a variable "
        "number of parameters across different inputs and models. "
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
        "gradient. Training may terminate earlier if the early-stopping criteria are met "
        "(controlled by loss_tol, patience, min_rel_improve, explode_factor, and warmup "
        "parameters hardcoded in main). Higher values allow more thorough optimisation at the "
        "cost of longer runtime."
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
        "The dataset is downloaded automatically to --data_path if not already present."
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
    
    parser.add_argument("--optimizer", type=str, default="lbfgs", choices=["lbfgs", "adam", "adamw"], help="Optimizer used for the reconstruction of dummy_data."
)
    
    args = parser.parse_args()

    # -------- Masking config --------
    MASK_MODE = args.mask_mode
    PREFIXES_NAME = args.prefixes
    PREFIXES = []
    PREFIX_LAYER_FRACS = {}
    for item in PREFIXES_NAME.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            prefix, frac = item.split(":", 1)
            prefix = prefix.strip()
            frac = float(frac.strip())
            PREFIXES.append(prefix)
            PREFIX_LAYER_FRACS[prefix] = frac
        else:
            PREFIXES.append(item)
    PREFIXES = tuple(PREFIXES)
    GRADSIZE_TOPK = args.gradsize_topk
    GRADSIZE_TOPFRAC = args.gradsize_topfrac
    GRADSIZE_THRESHOLD = args.gradsize_threshold
    GRADSIZE_METRIC = args.gradsize_metric
    METHODS = args.methods
    COMPUTE_JACOBIAN_RANK = args.compute_jacobian_rank
    JACOBIAN_MAX_ENTRIES = args.jacobian_max_entries
    JACOBIAN_SELECT_MODE = args.jacobian_select_mode
    TV_WEIGHT = args.tv_weight
    OPTIMIZER = args.optimizer
    lr = args.lr
    num_dummy = args.num_dummy
    Iteration = args.iteration
    num_exp = args.num_exp
    NETWORK_NAME = args.network
    dataset = args.dataset
    run_id = args.run_id
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    loss_tol = 1e-6
    patience = 100
    min_rel_improve = 1e-7
    explode_factor = 20.0
    warmup = 300
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

    best_psnr_idlg_all = []
    best_psnr_masked_all = []
    best_loss_idlg_all = []
    best_mse_idlg_all = []
    best_loss_masked_all = []
    best_mse_masked_all = []

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
        'run_id': args.run_id,
        'MASK_MODE': MASK_MODE,
        'PREFIXES': PREFIXES,
        'PREFIX_LAYER_FRACS': PREFIX_LAYER_FRACS,
        'GRADSIZE_TOPK': GRADSIZE_TOPK,
        'GRADSIZE_TOPFRAC': GRADSIZE_TOPFRAC,
        'GRADSIZE_THRESHOLD': GRADSIZE_THRESHOLD,
        'GRADSIZE_METRIC': GRADSIZE_METRIC,
        'NETWORK_NAME': NETWORK_NAME,
        'USE_INVERSEFED_IDLG': NETWORK_NAME.lower() == "resnet18",
        'METHODS': METHODS,
        'COMPUTE_JACOBIAN_RANK': COMPUTE_JACOBIAN_RANK,
        'JACOBIAN_MAX_ENTRIES': JACOBIAN_MAX_ENTRIES,
        'JACOBIAN_SELECT_MODE': JACOBIAN_SELECT_MODE,
        'TV_WEIGHT': TV_WEIGHT,
        'OPTIMIZER': OPTIMIZER,
        'run_id': run_id,
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
    unknowns = channel * shape_img[0] * shape_img[1]
    print(f"Input unknowns per image: {unknowns}")
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
        #print(f"Launching experiment {next_exp} on GPU {device_id}", flush=True)
        tqdm.write(f"Launching experiment {next_exp} on GPU {device_id}")
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

            if 'iDLG' in result['final_recon']:
                idlg_pil = tp(torch.from_numpy(result['final_recon']['iDLG'])[0])
            else:
                idlg_pil = gt_pil
            panel_idlg_pil.append(idlg_pil)

            if 'iDLG_masked' in result['final_recon']:
                masked_pil = tp(torch.from_numpy(result['final_recon']['iDLG_masked'])[0])
            else:
                masked_pil = gt_pil
            panel_masked_pil.append(masked_pil)

            if len(panel_gt_pil) == panel_block_size:
                save_recon_panel(
                    params, panel_gt_pil, panel_idlg_pil, panel_masked_pil,
                    save_path, panel_block_idx, dataset, mask_desc, timestamp_str, methods=METHODS
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
                #print(f"Launching experiment {next_exp} on GPU {finished_device}", flush=True)
                tqdm.write(f"Launching experiment {next_exp} on GPU {finished_device}")
                next_exp += 1

    # Final cleanup
    for p in active_processes.values():
        p.join()
    # save any remaining
    if len(panel_gt_pil) > 0:
        save_recon_panel(params, panel_gt_pil, panel_idlg_pil, panel_masked_pil,
                         save_path, panel_block_idx, dataset, mask_desc, timestamp_str, methods=METHODS)
    
    # -------- Compute statistics --------
    avg_psnr_idlg           = float(np.mean(psnr_idlg_all))             if len(psnr_idlg_all)       else float("nan")
    avg_psnr_masked         = float(np.mean(psnr_masked_all))           if len(psnr_masked_all)     else float("nan")
    std_psnr_idlg           = float(np.std(psnr_idlg_all)) if len(psnr_idlg_all) else float("nan")
    std_psnr_masked         = float(np.std(psnr_masked_all)) if len(psnr_masked_all) else float("nan")

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
    std_best_psnr_idlg         = float(np.std(best_psnr_idlg_all))          if best_psnr_idlg_all else float("nan")
    std_best_psnr_masked       = float(np.std(best_psnr_masked_all))        if best_psnr_masked_all else float("nan")

    avg_best_loss_idlg         = float(np.mean(best_loss_idlg_all))         if best_loss_idlg_all else float("nan")
    avg_best_mse_idlg          = float(np.mean(best_mse_idlg_all))          if best_mse_idlg_all else float("nan")
    avg_best_loss_masked       = float(np.mean(best_loss_masked_all))       if best_loss_masked_all else float("nan")
    avg_best_mse_masked        = float(np.mean(best_mse_masked_all))        if best_mse_masked_all else float("nan")

    med_best_loss_idlg         = float(np.median(best_loss_idlg_all))       if best_loss_idlg_all else float("nan")
    med_best_mse_idlg          = float(np.median(best_mse_idlg_all))        if best_mse_idlg_all else float("nan")
    med_best_loss_masked       = float(np.median(best_loss_masked_all))     if best_loss_masked_all else float("nan")
    med_best_mse_masked        = float(np.median(best_mse_masked_all))      if best_mse_masked_all else float("nan")

    csv_path = os.path.join(save_path, "exp_results.csv")
    file_exists = os.path.isfile(csv_path)

    common = {
        "timestamp": timestamp_str,
        "dataset": dataset,
        "network": NETWORK_NAME,
        
        "lr": lr,
        "iteration": Iteration,
        "num_exp": num_exp,
        "tv_weight": TV_WEIGHT,
        "optimizer": OPTIMIZER}
    
    grad_value = ""
    if MASK_MODE in [
        "gradsize_topfrac",
        "gradsize_topfrac_entries",
        "prefix_topfrac",
        "prefix_topfrac_entries",
        "prefix_topfrac_entries_layer",
    ]:
        grad_value = GRADSIZE_TOPFRAC
    elif MASK_MODE in [
        "gradsize_topk",
        "gradsize_topk_entries",
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
            # "med_final_loss": med_final_loss_idlg,
            # "avg_final_loss": avg_final_loss_idlg,
            # "med_final_mse": med_final_mse_idlg,
            # "avg_final_mse": avg_final_mse_idlg,
            # "avg_psnr": avg_psnr_idlg,
            # "std_psnr": std_psnr_idlg,
            "med_best_loss": round(med_best_loss_idlg,5),
            "avg_best_loss": round(avg_best_loss_idlg,5),
            "med_best_mse": round(med_best_mse_idlg,5),
            "avg_best_mse": round(avg_best_mse_idlg,5),
            "avg_best_psnr": round(avg_best_psnr_idlg,5),
            "std_best_psnr": round(std_best_psnr_idlg,5),
        })

    if METHODS in ["masked", "both"]:
        rows.append({
            "method": "iDLG_masked",
            **common,
            "mask_mode": MASK_MODE, #if MASK_MODE != 'prefix' else args.prefixes,
            "prefixes": args.prefixes if "prefix" in MASK_MODE else "",
            "grad_param": grad_value,
            # "med_final_loss": med_final_loss_masked,
            # "avg_final_loss": avg_final_loss_masked,
            # "med_final_mse": med_final_mse_masked,
            # "avg_final_mse": avg_final_mse_masked,
            # "avg_psnr": avg_psnr_masked,
            # "std_psnr": std_psnr_masked,
            "med_best_loss": round(med_best_loss_masked,5),
            "avg_best_loss": round(avg_best_loss_masked,5),
            "med_best_mse": round(med_best_mse_masked,5),
            "avg_best_mse": round(avg_best_mse_masked,5),
            "avg_best_psnr": round(avg_best_psnr_masked,5),
            "std_best_psnr": round(std_best_psnr_masked,5,)
        })

    fieldnames = ["method"] + [k for k in common.keys()] + [
    "mask_mode", "prefixes", "grad_param",
    # "med_final_loss", "avg_final_loss", "med_final_mse", "avg_final_mse", "avg_psnr", "std_psnr",
    "med_best_loss", "avg_best_loss", "med_best_mse", "avg_best_mse", "avg_best_psnr", "std_best_psnr"]

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
    print(f"Average PSNR iDLG: {avg_psnr_idlg:.4f} ± {std_psnr_idlg:.4f} dB | masked: {avg_psnr_masked:.4f} ± {std_psnr_masked:.4f} dB")
    print(f"Avg best loss iDLG: {avg_best_loss_idlg:.6f} | masked: {avg_best_loss_masked:.6f}")
    print(f"Avg best mse  iDLG: {avg_best_mse_idlg:.8f} | masked: {avg_best_mse_masked:.8f}")
    print(f"Median best loss iDLG: {med_best_loss_idlg:.6f} | masked: {med_best_loss_masked:.6f}")
    print(f"Median best mse  iDLG: {med_best_mse_idlg:.8f} | masked: {med_best_mse_masked:.8f}")
    print(f"Average best PSNR iDLG: {avg_best_psnr_idlg:.4f} ± {std_best_psnr_idlg:.4f} dB | masked: {avg_best_psnr_masked:.4f} ± {std_best_psnr_masked:.4f} dB")
if __name__ == '__main__':
    main()