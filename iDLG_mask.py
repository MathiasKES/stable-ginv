# iDLG_mask.py
import os
import sys
import torch
from torchvision import transforms
from datetime import datetime

import torch.multiprocessing as mp
from functions.experiment_results import (
    append_result_metrics,
    build_common_csv_fields,
    build_exp_result_rows,
    compute_aggregate_stats,
    create_metric_accumulators,
    empty_paired_report,
    grad_param_value,
    merge_restart_results,
    ordered_idlg_baseline_lists,
    paired_report_for_both,
    paired_report_for_masked,
    print_final_experiment_summary,
    print_result_summary,
    write_masking_sweep_mse_csv,
)
from functions.idlg_cli import parse_idlg_args
from functions.io_utils import (baseline_key_from_args,
    load_baseline_registry, save_baseline_registry, update_idlg_baseline,
    write_baseline_summary_csv, parse_prefixes_with_fracs, masked_key_from_args,
    load_masked_registry, save_masked_registry, update_masked_registry,
    append_csv_rows, resolve_storage_paths, safe_makedirs)
from functions.Dataset import load_dataset
from run_single_exp import run_single_experiment
from tqdm import tqdm

from functions.io_utils import setstdout


class ExperimentRunAborted(RuntimeError):
    """Raised after a worker failure has caused the run to be cancelled and cleaned up."""


def _load_visualization_helpers():
    """Load plotting helpers; fall back to no-op plotting when visualization imports fail."""
    try:
        from helper.visualization import (
            append_result_to_panel_buffers,
            create_panel_buffers,
            flush_recon_panel,
            save_recon_gif,
            save_restart_curve,
            save_restart_images,
        )
        return (
            append_result_to_panel_buffers,
            create_panel_buffers,
            flush_recon_panel,
            save_recon_gif,
            save_restart_curve,
            save_restart_images,
        )
    except Exception as exc:
        print(f"[WARNING] Visualization disabled because plotting libraries could not be imported: {exc}")
        print("[WARNING] Experiments will continue; use registry/CSV outputs to plot later.")

        def create_panel_buffers():
            return {
                "gt": [],
                "idlg": [],
                "masked": [],
                "psnr_idlg": [],
                "ssim_idlg": [],
                "mse_idlg": [],
                "psnr_masked": [],
                "ssim_masked": [],
                "mse_masked": [],
            }

        def append_result_to_panel_buffers(result, buffers, to_pil, warn_fn):
            return None

        def flush_recon_panel(params, buffers, panel_png_paths, save_dir, block_idx,
                              dataset, mask_desc, timestamp_str, methods):
            return block_idx

        def save_recon_gif(*args, **kwargs):
            return None

        def save_restart_curve(*args, **kwargs):
            return None

        def save_restart_images(*args, **kwargs):
            return None

        return (
            append_result_to_panel_buffers,
            create_panel_buffers,
            flush_recon_panel,
            save_recon_gif,
            save_restart_curve,
            save_restart_images,
        )


def main():
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = setstdout(ts=timestamp_str)
    args = parse_idlg_args(sys.argv)
    (
        append_result_to_panel_buffers,
        create_panel_buffers,
        flush_recon_panel,
        save_recon_gif,
        save_restart_curve,
        save_restart_images,
    ) = _load_visualization_helpers()

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

    # Set datapath
    root_path = '.'
    data_path, save_path = resolve_storage_paths(root_path)

    tp = transforms.Compose([transforms.ToPILImage()])

    safe_makedirs(data_path)
    safe_makedirs(save_path)
    
    # Set paths
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
    panel_png_paths = []
    panel_buffers = create_panel_buffers()
    metric_accumulators = create_metric_accumulators()

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
        'out_path': out_path,
    }

    # -------- Run experiments in parallel --------
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs")
    if num_gpus == 0:
        raise RuntimeError("No CUDA GPUs available.")
    
    # Set parallel worker method
    mp.set_start_method('spawn', force=True)
    mp.set_sharing_strategy('file_system')

    result_queue = mp.SimpleQueue()
    active_processes = {}
    all_results_by_idx = {}

    def _abort_run(result):
        idx = result.get('idx_net')
        dev = result.get('device_id')
        r_i = result.get('restart_idx')
        where = f"exp {idx}" + (f" restart {r_i}" if r_i is not None else "") + f" on GPU {dev}"
        print(f"\n[ABORT] {where} failed — cancelling entire run, no results will be saved.")
        print(result.get('traceback', result.get('error', '')))
        for proc in active_processes.values():
            if proc.is_alive():
                proc.terminate()
        for proc in active_processes.values():
            proc.join()
        raise ExperimentRunAborted(
            f"Run cancelled: {where} failed with: {result.get('error', 'unknown error')}"
        )

    parallel_restarts = num_exp < num_gpus and NUM_RESTARTS > 1

    # ---- _handle_result closure: accumulates lists, builds panel, prints ----
    def _handle_result(result):
        nonlocal panel_block_idx
        idx = result["idx_net"]
        all_results_by_idx[idx] = result

        append_result_metrics(result, metric_accumulators)
        append_result_to_panel_buffers(result, panel_buffers, tp, tqdm.write)
        if len(panel_buffers["gt"]) == panel_block_size:
            panel_block_idx = flush_recon_panel(
                params, panel_buffers, panel_png_paths, save_path, panel_block_idx,
                dataset, mask_desc, timestamp_str, METHODS,
            )

        print_result_summary(result)

    if parallel_restarts:
        print(f"[INFO] Parallel restart mode: {num_exp} exp × {NUM_RESTARTS} restarts across {num_gpus} GPUs")
        # Round-robin across images so all experiments get restarts interleaved —
        # prevents 3 GPUs idling while only one image's final restart is running.
        tasks = [(exp_i, r_i) for r_i in range(NUM_RESTARTS) for exp_i in range(num_exp)]
        total_tasks = len(tasks)
        restart_buf = {exp_i: {} for exp_i in range(num_exp)}
        next_task = 0
        completed_tasks = 0

        for device_id in range(min(num_gpus, total_tasks)):
            exp_i, r_i = tasks[next_task]
            task_cfg = {**config, 'SINGLE_RESTART_IDX': r_i}
            p = mp.Process(target=run_single_experiment,
                           args=(exp_i, device_id, dst, dataset, task_cfg, result_queue))
            p.start()
            active_processes[device_id] = p
            next_task += 1

        with tqdm(total=total_tasks, desc="Restart tasks", position=0) as pbar:
            while completed_tasks < total_tasks:
                result = result_queue.get()
                completed_tasks += 1
                pbar.update(1)
                finished_device = result['device_id']
                active_processes[finished_device].join()

                if result.get('error') is not None:
                    _abort_run(result)
                exp_i = result['idx_net']
                r_i = result.get('restart_idx', 0)
                restart_buf[exp_i][r_i] = result

                if next_task < total_tasks:
                    exp_i, r_i = tasks[next_task]
                    task_cfg = {**config, 'SINGLE_RESTART_IDX': r_i}
                    p = mp.Process(target=run_single_experiment,
                                   args=(exp_i, finished_device, dst, dataset, task_cfg, result_queue))
                    p.start()
                    active_processes[finished_device] = p
                    next_task += 1

        for p in active_processes.values():
            p.join()

        for exp_i in range(num_exp):
            if restart_buf[exp_i]:
                _handle_result(merge_restart_results(restart_buf[exp_i]))

        panel_block_idx = flush_recon_panel(
            params, panel_buffers, panel_png_paths, save_path, panel_block_idx,
            dataset, mask_desc, timestamp_str, METHODS,
        )

    else:
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
            next_exp += 1

        # Keep launching a new experiment whenever one finishes
        with tqdm(total=num_exp, desc="Experiments", position=0) as pbar:
            while completed < num_exp:
                result = result_queue.get()
                completed += 1
                pbar.update(1)
                pbar.set_postfix(last_exp=result["idx_net"], gpu=result["device_id"])

                finished_device = result['device_id']

                if result.get('error') is not None:
                    _abort_run(result)

                _handle_result(result)

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
        panel_block_idx = flush_recon_panel(
            params, panel_buffers, panel_png_paths, save_path, panel_block_idx,
            dataset, mask_desc, timestamp_str, METHODS,
        )

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

    # -------- Restart curve: CSV + plot --------
    psnr_per_restart_idlg_all = metric_accumulators["psnr_per_restart_idlg"]
    psnr_per_restart_masked_all = metric_accumulators["psnr_per_restart_masked"]
    mse_per_restart_idlg_all = metric_accumulators["mse_per_restart_idlg"]
    mse_per_restart_masked_all = metric_accumulators["mse_per_restart_masked"]
    if NUM_RESTARTS > 1 and (psnr_per_restart_idlg_all or psnr_per_restart_masked_all):
        save_restart_curve(
            save_path, timestamp_str, NUM_RESTARTS, NETWORK_NAME, dataset, MASK_MODE,
            psnr_per_restart_idlg_all, psnr_per_restart_masked_all,
            mse_per_restart_idlg_all, mse_per_restart_masked_all,
        )
        save_restart_images(
            save_path, timestamp_str, NUM_RESTARTS, NETWORK_NAME, dataset, MASK_MODE,
            num_exp, all_results_by_idx,
            psnr_per_restart_idlg_all, psnr_per_restart_masked_all,
        )

    # -------- Paired statistics --------
    baseline_key, comparable_args = baseline_key_from_args(args)
    paired_report = empty_paired_report()
    if METHODS == "both":
        paired_report = paired_report_for_both(all_results_by_idx, tqdm.write)

    elif METHODS == "masked":
        registry = load_baseline_registry(baseline_registry_path)

        if baseline_key not in registry:
            print("\nWARNING: No matching iDLG baseline found for this masked run.")
            print("Run the same command with --methods idlg first, using the same non-mask arguments.")
            print(f"Expected baseline key: {baseline_key}")
        else:
            paired_report = paired_report_for_masked(all_results_by_idx, registry[baseline_key], run_id)

    # -------- Compute statistics --------
    stats = compute_aggregate_stats(metric_accumulators)
    # -------- Save/update iDLG baseline registry --------
    if METHODS in ("idlg", "both"):
        ordered_best_psnr_idlg, ordered_best_mse_idlg, ordered_best_ssim_idlg = (
            ordered_idlg_baseline_lists(all_results_by_idx)
        )

        registry = load_baseline_registry(baseline_registry_path)

        update_idlg_baseline(
            registry,
            baseline_key,
            comparable_args,
            ordered_best_psnr_idlg,
            ordered_best_mse_idlg,
            ordered_best_ssim_idlg,
        )

        save_baseline_registry(baseline_registry_path, registry)
        write_baseline_summary_csv(baseline_summary_csv_path, registry)

        print("\nSaved iDLG baseline:")
        print(f"baseline_key: {baseline_key}")
        print(f"registry: {baseline_registry_path}")
        print(f"summary csv: {baseline_summary_csv_path}")

    write_sweep_mse_csv = MASK_MODE == "gradsize_topfrac_entries_layer" and METHODS in ["masked", "both"]
    masked_key = None
    masked_comparable_args = None

    if METHODS in ["masked", "both"] and metric_accumulators["best_psnr_masked"]:
        masked_registry = load_masked_registry(masked_registry_path)
        masked_key, masked_comparable_args = masked_key_from_args(args)
        update_masked_registry(
            masked_registry,
            masked_key,
            masked_comparable_args,
            metric_accumulators["best_psnr_masked"],
            metric_accumulators["best_mse_masked"],
            metric_accumulators["best_ssim_masked"],
        )
        save_masked_registry(masked_registry_path, masked_registry)
        print("\nSaved masked registry entry:")
        print(f"masked_key: {masked_key}")
        print(f"registry: {masked_registry_path}")

    csv_path = os.path.join(save_path, f"exp_results_{NETWORK_NAME}.csv")

    png_path_str = "|".join(panel_png_paths)
    common = build_common_csv_fields(
        timestamp_str, sys.argv, dataset, NETWORK_NAME, NETWORK_TRAINED,
        NUM_RESTARTS, lr, GAMMA, Iteration, num_exp, TV_WEIGHT, OPTIMIZER,
        MAX_ITERATION, HISTORY_SIZE,
    )
    grad_value = grad_param_value(MASK_MODE, GRADSIZE_TOPFRAC, GRADSIZE_TOPK)
    rows, fieldnames = build_exp_result_rows(
        METHODS, common, stats, paired_report, baseline_key, masked_key,
        MASK_MODE, args.prefixes, grad_value, png_path_str,
    )

    append_csv_rows(csv_path, rows, fieldnames)
    if write_sweep_mse_csv:
        write_masking_sweep_mse_csv(
            save_path, NETWORK_NAME, dataset, MASK_MODE, masked_key,
            masked_comparable_args, GRADSIZE_TOPFRAC, sys.argv,
        )

    print_final_experiment_summary(
        csv_path, GAMMA, MASK_MODE, GRADSIZE_TOPFRAC, GRADSIZE_TOPK,
        stats, paired_report, METHODS,
    )

    print("Job resource usage:")
    print("Max memory allocated:", torch.cuda.memory.max_memory_allocated() / (1024 ** 3), "GB")
    print("Max memory reserved:", torch.cuda.memory.max_memory_reserved() / (1024 ** 3), "GB")

    print("Memory summary:")
    print(torch.cuda.memory.memory_summary())

        
if __name__ == '__main__':
    main()
