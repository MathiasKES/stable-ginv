"""Golden test: baseline summary CSV is byte-stable across the Phase 4 registry split.

Guards write_baseline_summary_csv against accidental changes to columns, ordering,
float formatting, JSON-encoded list cells, or the dense/sparse SSIM handling.
"""
import os
import tempfile

from stable_ginv.registry import write_baseline_summary_csv
from tests.golden.helpers import load_or_regen


def _base_args(num_exp, run_id):
    return {
        "dataset": "cifar100",
        "network": "resnet18",
        "pretrained": False,
        "lr": 0.1,
        "gamma": 0.5,
        "grad_loss": "cos",
        "num_dummy": 1,
        "iteration": 5000,
        "num_exp": num_exp,
        "run_id": run_id,
        "tv_weight": 0.0,
        "optimizer": "signed_adamw",
        "num_restarts": 1,
        "max_iteration": 20,
        "history_size": 100,
    }


def _registry():
    # Three entries exercising: dense SSIM, sparse SSIM (best_ssim_by_run_id),
    # and an entry with no SSIM at all (empty list -> nan summary cells).
    return {
        "key_dense": {
            "args": _base_args(num_exp=3, run_id=0),
            "best_psnr_list": [10.5, 11.0, 9.5],
            "best_mse_list": [0.01, 0.02, 0.015],
            "best_ssim_list": [0.80, 0.85, 0.82],
        },
        "key_sparse": {
            "args": _base_args(num_exp=2, run_id=3),
            "best_psnr_list": [12.0, 13.0],
            "best_mse_list": [0.005, 0.004],
            "best_ssim_by_run_id": {"3": 0.90},
        },
        "key_nossim": {
            "args": _base_args(num_exp=1, run_id=7),
            "best_psnr_list": [8.0],
            "best_mse_list": [0.03],
        },
    }


def _produce():
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    try:
        write_baseline_summary_csv(path, _registry())
        with open(path) as f:
            return f.read()
    finally:
        os.remove(path)


def test_baseline_summary_csv_stable():
    golden = load_or_regen("baseline_summary_csv_golden.json", _produce)
    assert _produce() == golden
