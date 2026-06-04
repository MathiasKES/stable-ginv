"""Baseline registry summary CSV writer."""
import csv
import json

import numpy as np

from stable_ginv.io import safe_write
from stable_ginv.registry.entries import available_ssim_values


def write_baseline_summary_csv(path, registry):
    """Write a summary CSV of all baseline registry entries. Failures are logged, not raised."""
    fieldnames = [
        "baseline_key",
        "dataset",
        "network",
        "pretrained",
        "lr",
        "gamma",
        "grad_loss",
        "num_dummy",
        "iteration",
        "num_exp",
        "run_id",
        "tv_weight",
        "optimizer",
        "num_restarts",
        "max_iteration",
        "history_size",
        "avg_best_psnr",
        "std_best_psnr",
        "avg_best_mse",
        "avg_best_ssim",
        "std_best_ssim",
        "best_psnr_list",
        "best_mse_list",
        "best_ssim_list",
    ]

    rows = []
    for key, entry in registry.items():
        a = entry["args"]
        psnr = np.array(entry["best_psnr_list"], dtype=float)
        mse = np.array(entry["best_mse_list"], dtype=float)
        ssim_list = available_ssim_values(entry)
        ssim = np.array(ssim_list, dtype=float)

        rows.append({
            "baseline_key": key,
            **a,
            "avg_best_psnr": float(np.mean(psnr)) if len(psnr) else float("nan"),
            "std_best_psnr": float(np.std(psnr, ddof=1)) if len(psnr) > 1 else float("nan"),
            "avg_best_mse": float(np.mean(mse)) if len(mse) else float("nan"),
            "avg_best_ssim": float(np.mean(ssim)) if len(ssim) else float("nan"),
            "std_best_ssim": float(np.std(ssim, ddof=1)) if len(ssim) > 1 else float("nan"),
            "best_psnr_list": json.dumps(entry["best_psnr_list"]),
            "best_mse_list": json.dumps(entry["best_mse_list"]),
            "best_ssim_list": json.dumps(ssim_list),
        })

    def _write(f):
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return safe_write(path, _write, newline="")
