"""iDLG masked gradient-inversion experiment entry point.

Owns the full batch run: argument parsing, data loading, ExperimentConfig
construction, scheduling via BatchExperimentRunner, result aggregation,
registry/paired statistics, CSV output, and the final summary.

Run from the repository root with::

    python -m stable_ginv.cli.batch [args]
"""
import os
import sys
from datetime import datetime

import torch
from torchvision import transforms
from tqdm import tqdm

from functions.idlg_cli import parse_idlg_args
from functions.Dataset import load_dataset
from stable_ginv.config import ExperimentConfig
from stable_ginv.io import (
    append_csv_rows,
    parse_prefixes_with_fracs,
    resolve_storage_paths,
    safe_makedirs,
    setstdout,
)
from stable_ginv.registry import (
    baseline_key_from_args,
    find_registry_entry,
    load_baseline_registry,
    load_masked_registry,
    masked_key_from_args,
    save_baseline_registry,
    save_masked_registry,
    seed_registry_entry_from_fallback,
    update_idlg_baseline,
    update_masked_registry,
    write_baseline_summary_csv,
)
from stable_ginv.experiment.results import (
    ResultAggregator,
    build_common_csv_fields,
    build_exp_result_rows,
    empty_paired_report,
    grad_param_value,
    ordered_idlg_baseline_lists,
    ordered_masked_registry_lists,
    paired_report_for_both,
    paired_report_for_masked,
    print_final_experiment_summary,
    print_result_summary,
    write_masking_sweep_mse_csv,
)
from stable_ginv.experiment.runner import BatchExperimentRunner


def _load_visualization_helpers():
    """Load plotting helpers; fall back to no-op plotting when visualization imports fail."""
    try:
        from stable_ginv.viz import (
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
            """Return empty panel-buffer dict (no-op fallback when viz is unavailable)."""
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
            """No-op fallback: discard result when viz is unavailable."""
            return None

        def flush_recon_panel(params, buffers, panel_png_paths, save_dir, block_idx,
                              dataset, mask_desc, timestamp_str, methods):
            """No-op fallback: return block_idx unchanged when viz is unavailable."""
            return block_idx

        def save_recon_gif(*args, **kwargs):
            """No-op fallback: skip GIF saving when viz is unavailable."""
            return None

        def save_restart_curve(*args, **kwargs):
            """No-op fallback: skip restart-curve output when viz is unavailable."""
            return None

        def save_restart_images(*args, **kwargs):
            """No-op fallback: skip restart image grid when viz is unavailable."""
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
    """Run a batch of gradient-inversion experiments and write all outputs.

    Parses CLI arguments via ``parse_idlg_args``, loads the requested dataset,
    builds an :class:`~stable_ginv.config.ExperimentConfig`, and dispatches
    ``num_exp`` experiments through :class:`~stable_ginv.experiment.runner.BatchExperimentRunner`
    (GPU workers when CUDA is available, otherwise CPU).  As each experiment
    completes its result is accumulated into panel buffers; filled blocks are
    flushed to PNG reconstruction panels.

    After all experiments finish the function:

    * saves animated GIFs of reconstruction progress (when ``--save_gif`` is set);
    * writes a restart-curve CSV and PNG plot (when ``--num_restarts`` > 1);
    * updates the iDLG baseline registry and/or the masked registry (depending
      on ``--methods``);
    * computes paired statistics against a stored baseline when running in
      ``masked``-only mode;
    * appends a summary row to ``exp_results_<network>.csv``;
    * writes a masking-sweep MSE CSV for ``gradsize_topfrac_entries_layer`` runs;
    * prints a final experiment summary and GPU memory usage.
    """
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
    legacy_baseline_registry_path = os.path.join(baseline_dir, "idlg_baselines_registry.json")
    baseline_registry_path = os.path.join(baseline_dir, "idlg_baselines_registry_v2.json")
    baseline_summary_csv_path = os.path.join(baseline_dir, "idlg_baselines_summary_v2.csv")
    legacy_masked_registry_path = os.path.join(baseline_dir, "masked_registry.json")
    masked_registry_path = os.path.join(baseline_dir, "masked_registry_v2.json")

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
    aggregator = ResultAggregator()

    mask_desc = MASK_MODE
    params = {"num-exp": num_exp, "lr": lr, "batchsize": num_dummy, "iters": Iteration}

    config = ExperimentConfig(
        channel=channel,
        num_classes=num_classes,
        shape_img=shape_img,
        lr=lr,
        num_dummy=num_dummy,
        iteration=Iteration,
        run_id=run_id,
        mask_mode=MASK_MODE,
        prefixes=PREFIXES,
        prefix_layer_fracs=PREFIX_LAYER_FRACS,
        gradsize_topk=GRADSIZE_TOPK,
        gradsize_topfrac=GRADSIZE_TOPFRAC,
        gradsize_metric=GRADSIZE_METRIC,
        grad_loss=GRAD_LOSS,
        gamma=GAMMA,
        network_name=NETWORK_NAME,
        methods=METHODS,
        compute_jacobian_rank=COMPUTE_JACOBIAN_RANK,
        jacobian_max_entries=JACOBIAN_MAX_ENTRIES,
        jacobian_select_mode=JACOBIAN_SELECT_MODE,
        tv_weight=TV_WEIGHT,
        optimizer=OPTIMIZER,
        num_restarts=NUM_RESTARTS,
        max_iteration=MAX_ITERATION,
        history_size=HISTORY_SIZE,
        network_trained=NETWORK_TRAINED,
        save_gif=SAVE_GIF,
        frame_interval=FRAME_INTERVAL,
        out_path=out_path,
    )

    # -------- Run experiments in parallel --------
    all_results_by_idx = {}

    # ---- _handle_result callback: accumulates lists, builds panel, prints ----
    def _handle_result(result):
        nonlocal panel_block_idx
        idx = result["idx_net"]
        all_results_by_idx[idx] = result

        aggregator.append(result)
        append_result_to_panel_buffers(result, panel_buffers, tp, tqdm.write)
        if len(panel_buffers["gt"]) == panel_block_size:
            panel_block_idx = flush_recon_panel(
                params, panel_buffers, panel_png_paths, save_path, panel_block_idx,
                dataset, mask_desc, timestamp_str, METHODS,
            )

        print_result_summary(result)

    runner = BatchExperimentRunner(config, dst, dataset, num_exp, NUM_RESTARTS)
    runner.run(_handle_result)

    # Flush any remaining buffered panels (both scheduling branches ended with this).
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
    psnr_per_restart_idlg_all = aggregator.accumulators["psnr_per_restart_idlg"]
    psnr_per_restart_masked_all = aggregator.accumulators["psnr_per_restart_masked"]
    mse_per_restart_idlg_all = aggregator.accumulators["mse_per_restart_idlg"]
    mse_per_restart_masked_all = aggregator.accumulators["mse_per_restart_masked"]
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

    baseline_key, comparable_args = baseline_key_from_args(args)
    masked_key = None
    masked_comparable_args = None

    # Save completed masked metrics before optional paired statistics.
    if METHODS in ["masked", "both"] and aggregator.accumulators["best_psnr_masked"]:
        ordered_best_psnr_masked, ordered_best_mse_masked, ordered_best_ssim_masked = (
            ordered_masked_registry_lists(all_results_by_idx)
        )
        masked_registry = load_masked_registry(masked_registry_path)
        legacy_masked_registry = load_masked_registry(legacy_masked_registry_path)
        masked_key, masked_comparable_args = masked_key_from_args(args)
        seed_registry_entry_from_fallback(
            masked_registry, legacy_masked_registry, masked_key, masked_comparable_args
        )
        update_masked_registry(
            masked_registry,
            masked_key,
            masked_comparable_args,
            ordered_best_psnr_masked,
            ordered_best_mse_masked,
            ordered_best_ssim_masked,
        )
        save_masked_registry(masked_registry_path, masked_registry)
        print("\nSaved masked registry entry:")
        print(f"masked_key: {masked_key}")
        print(f"registry: {masked_registry_path}")

    # -------- Paired statistics --------
    paired_report = empty_paired_report()
    if METHODS == "both":
        paired_report = paired_report_for_both(all_results_by_idx, tqdm.write)

    elif METHODS == "masked":
        registry = load_baseline_registry(baseline_registry_path)
        legacy_registry = load_baseline_registry(legacy_baseline_registry_path)
        try:
            stored_baseline_key, baseline_entry = find_registry_entry(
                registry, baseline_key, comparable_args
            )
            if baseline_entry is None:
                stored_baseline_key, baseline_entry = find_registry_entry(
                    legacy_registry, baseline_key, comparable_args
                )

            if baseline_entry is None:
                print("\nWARNING: No matching iDLG baseline found for this masked run.")
                print("Run the same command with --methods idlg first, using the same non-mask arguments.")
                print(f"Expected baseline key: {baseline_key}")
            else:
                if stored_baseline_key != baseline_key:
                    print(f"\nLoaded legacy iDLG baseline entry: {stored_baseline_key}")
                paired_report = paired_report_for_masked(
                    all_results_by_idx, baseline_entry, run_id
                )
        except (KeyError, TypeError, ValueError) as exc:
            print(f"\nWARNING: Skipping paired statistics for this masked run: {exc}")

    # -------- Compute statistics --------
    stats = aggregator.aggregate_stats()
    # -------- Save/update iDLG baseline registry --------
    if METHODS in ("idlg", "both"):
        ordered_best_psnr_idlg, ordered_best_mse_idlg, ordered_best_ssim_idlg = (
            ordered_idlg_baseline_lists(all_results_by_idx)
        )

        registry = load_baseline_registry(baseline_registry_path)
        legacy_registry = load_baseline_registry(legacy_baseline_registry_path)
        seed_registry_entry_from_fallback(registry, legacy_registry, baseline_key, comparable_args)

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

    if torch.cuda.is_available():
        print("Job resource usage:")
        print("Max memory allocated:", torch.cuda.memory.max_memory_allocated() / (1024 ** 3), "GB")
        print("Max memory reserved:", torch.cuda.memory.max_memory_reserved() / (1024 ** 3), "GB")

        print("Memory summary:")
        print(torch.cuda.memory.memory_summary())


if __name__ == "__main__":
    main()
