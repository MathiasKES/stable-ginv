import csv

from helper.plot_paired_masking_violin import _load_paired_rows, _paired_rows


def _entry(run_id, psnr, mse, ssim):
    return {
        "args": {"run_id": run_id},
        "best_psnr_list": psnr,
        "best_mse_list": mse,
        "best_ssim_list": ssim,
    }


def test_paired_rows_aligns_registry_entries_by_run_id_overlap():
    baseline = _entry(
        run_id=0,
        psnr=[10.0, 20.0, 30.0],
        mse=[0.3, 0.2, 0.1],
        ssim=[0.1, 0.2, 0.3],
    )
    masked = _entry(
        run_id=1,
        psnr=[21.0, 31.0, 41.0],
        mse=[0.19, 0.09, 0.04],
        ssim=[0.21, 0.31, 0.41],
    )

    rows = _paired_rows(baseline, masked)

    assert [row["run_id"] for row in rows] == [1, 2]
    assert [row["baseline_index"] for row in rows] == [1, 2]
    assert [row["masked_index"] for row in rows] == [0, 1]
    assert [row["delta_psnr"] for row in rows] == [1.0, 1.0]


def test_load_paired_rows_parses_corrected_csv(tmp_path):
    path = tmp_path / "corrected.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "run_id", "sample_index", "sample_number", "baseline_index", "masked_index",
            "baseline_psnr", "masked_psnr", "delta_psnr",
            "baseline_mse", "masked_mse", "delta_mse",
            "baseline_ssim", "masked_ssim", "delta_ssim",
        ])
        writer.writeheader()
        writer.writerow({
            "run_id": 2, "sample_index": 2, "sample_number": 3,
            "baseline_index": 2, "masked_index": 1,
            "baseline_psnr": 10, "masked_psnr": 11, "delta_psnr": 1,
            "baseline_mse": 0.2, "masked_mse": 0.1, "delta_mse": -0.1,
            "baseline_ssim": 0.3, "masked_ssim": 0.4, "delta_ssim": 0.1,
        })

    rows = _load_paired_rows(path)

    assert rows[0]["masked_index"] == 1
    assert rows[0]["delta_psnr"] == 1.0
