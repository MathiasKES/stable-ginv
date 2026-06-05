"""Golden test: the restart-curve CSV (with gain columns) is byte-stable across the Phase 7 split.

save_restart_curve writes a CSV; for num_restarts>=3 it rewrites that CSV with paired
restart-gain columns (gain/ci_low/ci_high/normality for PSNR and MSE, per k in {3,5,10}
that is <= num_restarts). This locks num_restarts=5 (gain_ks=[3,5]) with three experiments
so paired_t_ci and the Shapiro normality string both have >=3 samples. The deterministic
inputs make the CSV bytes reproducible. The PNG side output is not golden-locked (matplotlib
binaries are not byte-stable); only the CSV file content is asserted.
"""
import os
import shutil
import tempfile

import matplotlib

matplotlib.use("Agg")  # headless: save_restart_curve also writes a PNG

from stable_ginv.viz import save_restart_curve
from tests.golden.helpers import load_or_regen


def _produce():
    psnr_idlg = [
        [10.0, 14.0, 17.0, 19.0, 20.0],
        [8.0, 12.0, 15.0, 17.0, 18.0],
        [9.0, 13.0, 16.0, 18.0, 19.0],
    ]
    psnr_masked = [
        [7.0, 10.0, 12.0, 13.0, 14.0],
        [6.0, 9.0, 11.0, 12.0, 13.0],
        [5.0, 8.0, 10.0, 11.0, 12.0],
    ]
    mse_idlg = [
        [0.20, 0.12, 0.08, 0.05, 0.04],
        [0.25, 0.15, 0.10, 0.07, 0.05],
        [0.22, 0.13, 0.09, 0.06, 0.045],
    ]
    mse_masked = [
        [0.30, 0.22, 0.18, 0.15, 0.13],
        [0.35, 0.26, 0.21, 0.18, 0.16],
        [0.32, 0.24, 0.19, 0.16, 0.14],
    ]
    workdir = tempfile.mkdtemp()
    try:
        csv_path, _png_path = save_restart_curve(
            save_dir=workdir,
            timestamp_str="GOLDEN",
            num_restarts=5,
            network_name="LeNet",
            dataset="MNIST",
            mask_mode="gradsize_topfrac_entries_layer",
            psnr_per_restart_idlg_all=psnr_idlg,
            psnr_per_restart_masked_all=psnr_masked,
            mse_per_restart_idlg_all=mse_idlg,
            mse_per_restart_masked_all=mse_masked,
        )
        with open(csv_path) as f:
            return f.read()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_restart_curve_csv_with_gains_stable():
    golden = load_or_regen("restart_curve_csv_golden.json", _produce)
    assert _produce() == golden
