"""BatchExperimentRunner: GPU scheduling and multiprocessing for the experiment sweep.

Extracted verbatim from iDLG_mask.py (Phase 6). Owns worker scheduling, output
ordering (per-experiment, and round-robin restart interleaving when experiments are
fewer than GPUs), restart merging, and abort-on-worker-failure. Reconstruction
numerics live in stable_ginv.recon; result aggregation/CSV in
stable_ginv.experiment.results. The caller supplies a `handle_result` callback that
receives each completed experiment result in order.
"""
import dataclasses

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from stable_ginv.recon import run_single_experiment
from stable_ginv.experiment.results import RestartSelector


class ExperimentRunAborted(RuntimeError):
    """Raised after a worker failure has caused the run to be cancelled and cleaned up."""


class BatchExperimentRunner:
    """Schedules per-experiment reconstruction workers across the available GPUs."""

    def __init__(self, config, dst, dataset, num_exp, num_restarts):
        self.config = config
        self.dst = dst
        self.dataset = dataset
        self.num_exp = num_exp
        self.num_restarts = num_restarts
        self.active_processes = {}

    def _abort_run(self, result):
        idx = result.get('idx_net')
        dev = result.get('device_id')
        r_i = result.get('restart_idx')
        where = f"exp {idx}" + (f" restart {r_i}" if r_i is not None else "") + f" on GPU {dev}"
        print(f"\n[ABORT] {where} failed — cancelling entire run, no results will be saved.")
        print(result.get('traceback', result.get('error', '')))
        for proc in self.active_processes.values():
            if proc.is_alive():
                proc.terminate()
        for proc in self.active_processes.values():
            proc.join()
        raise ExperimentRunAborted(
            f"Run cancelled: {where} failed with: {result.get('error', 'unknown error')}"
        )

    def run(self, handle_result):
        config = self.config
        dst = self.dst
        dataset = self.dataset
        num_exp = self.num_exp
        NUM_RESTARTS = self.num_restarts

        # -------- Run experiments in parallel --------
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            print("[WARNING] No CUDA GPU detected — falling back to CPU with a single "
                  "worker. This is very slow and intended only for testing; a GPU is "
                  "strongly recommended for real experiments.")
            num_workers = 1
        else:
            print(f"Using {num_gpus} GPUs")
            num_workers = num_gpus

        # Set parallel worker method
        mp.set_start_method('spawn', force=True)
        mp.set_sharing_strategy('file_system')

        result_queue = mp.SimpleQueue()
        active_processes = self.active_processes

        parallel_restarts = num_exp < num_gpus and NUM_RESTARTS > 1

        if parallel_restarts:
            print(f"[INFO] Parallel restart mode: {num_exp} exp × {NUM_RESTARTS} restarts across {num_gpus} GPUs")
            # Round-robin across images so all experiments get restarts interleaved —
            # prevents 3 GPUs idling while only one image's final restart is running.
            tasks = [(exp_i, r_i) for r_i in range(NUM_RESTARTS) for exp_i in range(num_exp)]
            total_tasks = len(tasks)
            restart_buf = {exp_i: {} for exp_i in range(num_exp)}
            next_task = 0
            completed_tasks = 0

            for device_id in range(min(num_workers, total_tasks)):
                exp_i, r_i = tasks[next_task]
                task_cfg = dataclasses.replace(config, single_restart_idx=r_i)
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
                        self._abort_run(result)
                    exp_i = result['idx_net']
                    r_i = result.get('restart_idx', 0)
                    restart_buf[exp_i][r_i] = result

                    if next_task < total_tasks:
                        exp_i, r_i = tasks[next_task]
                        task_cfg = dataclasses.replace(config, single_restart_idx=r_i)
                        p = mp.Process(target=run_single_experiment,
                                       args=(exp_i, finished_device, dst, dataset, task_cfg, result_queue))
                        p.start()
                        active_processes[finished_device] = p
                        next_task += 1

            for p in active_processes.values():
                p.join()

            for exp_i in range(num_exp):
                if restart_buf[exp_i]:
                    handle_result(RestartSelector.merge(restart_buf[exp_i]))

        else:
            next_exp = 0
            completed = 0

            # Start one experiment per worker initially (one per GPU, or one on CPU)
            for device_id in range(min(num_workers, num_exp)):
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
                        self._abort_run(result)

                    handle_result(result)

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
