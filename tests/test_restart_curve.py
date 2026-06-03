import csv
import os

import matplotlib

matplotlib.use("Agg")  # headless: save_restart_curve writes a PNG

from helper.visualization import save_restart_curve


def _read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_restart_curve_csv_values(tmp_path):
    # num_restarts=2 keeps gain_ks empty (gains need k>=3), so the CSV is not
    # rewritten with gain columns -- this isolates the per-restart mean/std
    # columns derived from _restart_stats.
    #
    # idlg cols -> means 9.0/17.0, stds |2|/sqrt2=1.41421, |6|/sqrt2=4.24264
    # masked cols -> means 8.0/15.0, stds |2|/sqrt2=1.41421, |8|/sqrt2=5.65685
    idlg = [[10.0, 20.0], [8.0, 14.0]]
    masked = [[9.0, 11.0], [7.0, 19.0]]
    csv_path, png_path = save_restart_curve(
        save_dir=str(tmp_path),
        timestamp_str="TEST",
        num_restarts=2,
        network_name="LeNet",
        dataset="MNIST",
        mask_mode="gradsize_topfrac_entries_layer",
        psnr_per_restart_idlg_all=idlg,
        psnr_per_restart_masked_all=masked,
        mse_per_restart_idlg_all=[[0.1, 0.05], [0.2, 0.08]],
        mse_per_restart_masked_all=[[0.12, 0.07], [0.22, 0.04]],
    )

    assert os.path.isfile(csv_path)
    rows = _read_rows(csv_path)
    assert [r["num_restarts"] for r in rows] == ["1", "2"]
    assert rows[0] == {
        "num_restarts": "1",
        "mean_psnr_idlg": "9.0",
        "std_psnr_idlg": "1.41421",
        "mean_psnr_masked": "8.0",
        "std_psnr_masked": "1.41421",
    }
    assert rows[1] == {
        "num_restarts": "2",
        "mean_psnr_idlg": "17.0",
        "std_psnr_idlg": "4.24264",
        "mean_psnr_masked": "15.0",
        "std_psnr_masked": "5.65685",
    }


def test_restart_curve_single_method_omits_masked_columns(tmp_path):
    csv_path, _ = save_restart_curve(
        save_dir=str(tmp_path),
        timestamp_str="TEST2",
        num_restarts=2,
        network_name="LeNet",
        dataset="MNIST",
        mask_mode="none",
        psnr_per_restart_idlg_all=[[10.0, 20.0], [8.0, 14.0]],
        psnr_per_restart_masked_all=[],
        mse_per_restart_idlg_all=[[0.1, 0.05], [0.2, 0.08]],
        mse_per_restart_masked_all=[],
    )
    rows = _read_rows(csv_path)
    assert set(rows[0].keys()) == {"num_restarts", "mean_psnr_idlg", "std_psnr_idlg"}
    assert rows[0]["mean_psnr_idlg"] == "9.0"
    assert rows[1]["mean_psnr_idlg"] == "17.0"
