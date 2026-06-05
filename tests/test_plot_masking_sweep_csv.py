from stable_ginv.viz.plot_masking_sweep_csv import _baseline_row


def _baseline_entry(run_id, num_exp, mses):
    return {
        "args": {
            "dataset": "cifar100",
            "network": "LeNet",
            "pretrained": False,
            "lr": 0.1,
            "gamma": 0.5,
            "grad_loss": "cos",
            "num_dummy": 1,
            "iteration": 1000,
            "tv_weight": 0.0,
            "optimizer": "lbfgs",
            "num_restarts": 1,
            "max_iteration": 20,
            "history_size": 100,
            "num_exp": num_exp,
            "run_id": run_id,
        },
        "best_psnr_list": [1.0] * len(mses),
        "best_mse_list": mses,
    }


def test_baseline_row_uses_masked_sample_range_to_select_legacy_entry():
    rows = [{
        "args": {
            **_baseline_entry(0, 3, [0.1, 0.2, 0.3])["args"],
            "mask_mode": "gradsize_topfrac_entries_layer",
            "gradsize_topfrac": 0.5,
        },
    }]
    baseline_registry = {
        "three-samples": _baseline_entry(0, 3, [0.1, 0.2, 0.3]),
        "five-samples": _baseline_entry(0, 5, [0.1, 0.2, 0.3, 0.4, 0.5]),
    }

    baseline = _baseline_row(rows, baseline_registry)

    assert baseline["baseline_key"] == "three-samples"
    assert baseline["per_sample_mse"] == [0.1, 0.2, 0.3]
