"""Golden test: per-run experiment_results CSV rows are byte-stable across the Phase 6 split.

Locks build_common_csv_fields + build_exp_result_rows + append_csv_rows end-to-end:
column names, column order, the iDLG/masked row construction, float rounding, and the
job_id/device/Run-by environment fields. methods='both' exercises both output rows.
The env vars and timestamp/argv are fixed so the bytes are deterministic.
"""
import os
import tempfile

from stable_ginv.experiment.results import (
    build_common_csv_fields,
    build_exp_result_rows,
)
from stable_ginv.io import append_csv_rows
from tests.golden.helpers import load_or_regen


# Only the keys build_exp_result_rows actually reads from `stats`.
_STATS = {
    "med_best_loss_idlg": 0.123456789,
    "avg_best_loss_idlg": 0.234567891,
    "med_best_mse_idlg": 0.000123456789,
    "avg_best_mse_idlg": 0.000234567891,
    "avg_best_psnr_idlg": 21.111111,
    "std_best_psnr_idlg": 1.222222,
    "avg_best_ssim_idlg": 0.811111,
    "std_best_ssim_idlg": 0.022222,
    "med_best_loss_masked": 0.323456789,
    "avg_best_loss_masked": 0.434567891,
    "med_best_mse_masked": 0.000323456789,
    "avg_best_mse_masked": 0.000434567891,
    "avg_best_psnr_masked": 18.999999,
    "std_best_psnr_masked": 2.333333,
    "avg_best_ssim_masked": 0.733333,
    "std_best_ssim_masked": 0.044444,
}

# Only the *_str keys build_exp_result_rows reads from the paired report.
_PAIRED = {
    "mse_ci_str": "[-0.0002, 0.0004]",
    "psnr_ci_str": "[-3.0, -1.2]",
    "ssim_ci_str": "[-0.10, -0.06]",
    "mse_significant_str": "yes",
    "psnr_significant_str": "yes",
    "ssim_significant_str": "no",
    "psnr_normality_str": "normal",
    "mse_normality_str": "normal",
    "ssim_normality_str": "non-normal",
}


def _produce():
    env_keys = ("LSB_JOBID", "LSB_INTERACTIVE", "USER", "LSB_QUEUE")
    saved = {k: os.environ.get(k) for k in env_keys}
    os.environ["LSB_JOBID"] = "123456"
    os.environ.pop("LSB_INTERACTIVE", None)
    os.environ["USER"] = "golden_user"
    os.environ["LSB_QUEUE"] = "gpuv100"
    try:
        common = build_common_csv_fields(
            "20260101_000000",
            ["iDLG_mask.py", "--methods", "both", "--network", "LeNet"],
            "MNIST", "LeNet", False,
            1, 1.0, 0.5, 10, 2, 0.0, "lbfgs", 20, 100,
        )
        rows, fieldnames = build_exp_result_rows(
            "both", common, _STATS, _PAIRED,
            "baselinekey123", "maskedkey456",
            "gradsize_topfrac", "conv1:1.0,fc:1.0", 0.5, "panelA.png|panelB.png",
        )
        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        try:
            append_csv_rows(path, rows, fieldnames)
            with open(path) as f:
                return f.read()
        finally:
            os.remove(path)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_experiment_results_csv_rows_stable():
    golden = load_or_regen("experiment_results_csv_golden.json", _produce)
    assert _produce() == golden
