from types import SimpleNamespace

import pytest

from functions.experiment_results import ordered_masked_registry_lists, paired_report_for_masked
from functions.io_utils import (
    baseline_key_from_args,
    find_registry_entry,
    masked_key_from_args,
    seed_registry_entry_from_fallback,
    update_idlg_baseline,
    update_masked_registry,
)


def _args(run_id, num_exp):
    return SimpleNamespace(
        dataset="cifar100",
        network="resnet18",
        pretrained=False,
        lr=1.0,
        gamma=0.5,
        grad_loss="cos",
        num_dummy=1,
        iteration=1000,
        num_exp=num_exp,
        run_id=run_id,
        tv_weight=0.0,
        optimizer="lbfgs",
        num_restarts=1,
        max_iteration=20,
        history_size=100,
        mask_mode="gradsize_topfrac_entries_layer",
        gradsize_topk=20,
        gradsize_topfrac=0.5,
        gradsize_metric="l2",
        prefixes="",
    )


def test_masked_key_carries_fc_forced_marker_and_isolates_legacy_entries():
    """New masked runs are tagged fc_forced=False and must not match/append to
    pre-change registry entries (whose stored args lack the marker)."""
    key, comparable = masked_key_from_args(_args(run_id=0, num_exp=3))
    assert comparable["fc_forced"] is False

    # A legacy entry produced before the FC change: same config, no marker.
    legacy_args = {k: v for k, v in comparable.items() if k != "fc_forced"}
    legacy_registry = {"legacy_hash": {"args": legacy_args}}

    stored_key, entry = find_registry_entry(legacy_registry, key, comparable)
    assert stored_key is None and entry is None


def test_baseline_registry_appends_contiguous_sample_ranges():
    registry = {}
    first_key, first_args = baseline_key_from_args(_args(run_id=0, num_exp=3))
    second_key, second_args = baseline_key_from_args(_args(run_id=3, num_exp=2))

    assert first_key == second_key
    update_idlg_baseline(registry, first_key, first_args, [1, 2, 3], [4, 5, 6], [7, 8, 9])
    entry = update_idlg_baseline(registry, second_key, second_args, [10, 11], [12, 13], [14, 15])

    assert entry["args"]["run_id"] == 0
    assert entry["args"]["num_exp"] == 5
    assert entry["best_psnr_list"] == [1.0, 2.0, 3.0, 10.0, 11.0]
    assert entry["best_mse_list"] == [4.0, 5.0, 6.0, 12.0, 13.0]
    assert entry["best_ssim_list"] == [7.0, 8.0, 9.0, 14.0, 15.0]


def test_masked_registry_prepends_earlier_contiguous_sample_range():
    registry = {}
    key, later_args = masked_key_from_args(_args(run_id=30, num_exp=2))
    _, earlier_args = masked_key_from_args(_args(run_id=0, num_exp=30))

    update_masked_registry(registry, key, later_args, [30, 31], [130, 131], [230, 231])
    entry = update_masked_registry(
        registry,
        key,
        earlier_args,
        list(range(30)),
        list(range(100, 130)),
        list(range(200, 230)),
    )

    assert entry["args"]["run_id"] == 0
    assert entry["args"]["num_exp"] == 32
    assert entry["best_psnr_list"] == [float(v) for v in range(32)]
    assert entry["best_mse_list"] == [float(v) for v in range(100, 132)]
    assert entry["best_ssim_list"] == [float(v) for v in range(200, 232)]


def test_masked_registry_appends_only_unseen_suffix_from_overlapping_range():
    registry = {}
    key, first_args = masked_key_from_args(_args(run_id=0, num_exp=3))
    _, overlapping_args = masked_key_from_args(_args(run_id=2, num_exp=2))

    update_masked_registry(registry, key, first_args, [1, 2, 3], [4, 5, 6], [7, 8, 9])
    entry = update_masked_registry(
        registry, key, overlapping_args, [10, 11], [12, 13], [14, 15]
    )

    assert entry["args"]["num_exp"] == 4
    assert entry["best_psnr_list"] == [1.0, 2.0, 3.0, 11.0]
    assert entry["best_mse_list"] == [4.0, 5.0, 6.0, 13.0]
    assert entry["best_ssim_list"] == [7.0, 8.0, 9.0, 15.0]


def test_registry_copies_legacy_key_before_prepending_without_editing_old_entry():
    registry = {}
    key, later_args = baseline_key_from_args(_args(run_id=30, num_exp=2))
    _, earlier_args = baseline_key_from_args(_args(run_id=0, num_exp=30))
    legacy_entry = {
        "args": later_args,
        "best_psnr_list": [30.0, 31.0],
        "best_mse_list": [130.0, 131.0],
        "best_ssim_list": [230.0, 231.0],
    }
    registry["legacy-key"] = legacy_entry

    update_idlg_baseline(
        registry,
        key,
        earlier_args,
        list(range(30)),
        list(range(100, 130)),
        list(range(200, 230)),
    )

    assert registry["legacy-key"] == legacy_entry
    assert registry[key]["args"]["run_id"] == 0
    assert registry[key]["args"]["num_exp"] == 32
    assert registry[key]["best_psnr_list"] == [float(v) for v in range(32)]


def test_registry_overwrites_exact_existing_range_with_newest_metrics():
    registry = {}
    key, args = baseline_key_from_args(_args(run_id=0, num_exp=3))
    update_idlg_baseline(registry, key, args, [1, 2, 3], [4, 5, 6], [7, 8, 9])

    entry = update_idlg_baseline(registry, key, args, [10, 11, 12], [13, 14, 15], [16, 17, 18])

    assert entry["args"]["num_exp"] == 3
    assert entry["best_psnr_list"] == [10.0, 11.0, 12.0]
    assert entry["best_mse_list"] == [13.0, 14.0, 15.0]
    assert entry["best_ssim_list"] == [16.0, 17.0, 18.0]


def test_registry_overwrites_contained_subrange_with_newest_metrics():
    registry = {}
    key, first_args = baseline_key_from_args(_args(run_id=0, num_exp=5))
    _, rerun_args = baseline_key_from_args(_args(run_id=2, num_exp=2))
    update_idlg_baseline(registry, key, first_args, [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15])

    entry = update_idlg_baseline(registry, key, rerun_args, [20, 21], [22, 23], [24, 25])

    assert entry["args"]["num_exp"] == 5
    assert entry["best_psnr_list"] == [1.0, 2.0, 20.0, 21.0, 5.0]
    assert entry["best_mse_list"] == [6.0, 7.0, 22.0, 23.0, 10.0]
    assert entry["best_ssim_list"] == [11.0, 12.0, 24.0, 25.0, 15.0]


def test_registry_removes_stale_ssim_when_newest_range_has_no_ssim():
    registry = {}
    key, first_args = baseline_key_from_args(_args(run_id=0, num_exp=3))
    _, rerun_args = baseline_key_from_args(_args(run_id=1, num_exp=1))
    update_idlg_baseline(registry, key, first_args, [1, 2, 3], [4, 5, 6], [7, 8, 9])

    entry = update_idlg_baseline(registry, key, rerun_args, [10], [11], [])

    assert "best_ssim_list" not in entry
    assert entry["best_ssim_by_run_id"] == {"0": 7.0, "2": 9.0}


def test_registry_copies_legacy_key_before_append_without_editing_old_entry():
    registry = {}
    key, first_args = baseline_key_from_args(_args(run_id=0, num_exp=3))
    _, second_args = baseline_key_from_args(_args(run_id=3, num_exp=2))
    legacy_entry = {
        "args": first_args,
        "best_psnr_list": [1.0, 2.0, 3.0],
        "best_mse_list": [4.0, 5.0, 6.0],
        "best_ssim_list": [7.0, 8.0, 9.0],
    }
    registry["legacy-key"] = legacy_entry

    update_idlg_baseline(registry, key, second_args, [10, 11], [12, 13], [14, 15])

    assert registry["legacy-key"] == legacy_entry
    assert registry[key]["args"]["num_exp"] == 5
    assert registry[key]["best_psnr_list"] == [1.0, 2.0, 3.0, 10.0, 11.0]


def test_registry_seeds_v2_from_read_only_legacy_entry():
    writable_registry = {}
    key, args = baseline_key_from_args(_args(run_id=3, num_exp=2))
    legacy_entry = {
        "args": baseline_key_from_args(_args(run_id=0, num_exp=3))[1],
        "best_psnr_list": [1.0, 2.0, 3.0],
        "best_mse_list": [4.0, 5.0, 6.0],
    }
    legacy_registry = {"legacy-key": legacy_entry}

    seed_registry_entry_from_fallback(writable_registry, legacy_registry, key, args)
    update_idlg_baseline(writable_registry, key, args, [10, 11], [12, 13], [14, 15])

    assert legacy_registry == {"legacy-key": legacy_entry}
    assert list(writable_registry) == [key]
    assert writable_registry[key]["best_psnr_list"] == [1.0, 2.0, 3.0, 10.0, 11.0]


def test_legacy_lookup_prefers_exact_requested_range():
    key, args = baseline_key_from_args(_args(run_id=0, num_exp=3))
    registry = {
        "small": {
            "args": baseline_key_from_args(_args(run_id=0, num_exp=3))[1],
            "best_psnr_list": [1.0, 2.0, 3.0],
        },
        "large": {
            "args": baseline_key_from_args(_args(run_id=0, num_exp=5))[1],
            "best_psnr_list": [1.0, 2.0, 3.0, 4.0, 5.0],
        },
    }

    stored_key, _ = find_registry_entry(registry, key, args)

    assert stored_key == "small"


def test_legacy_lookup_prefers_smallest_range_containing_requested_subset():
    key, args = baseline_key_from_args(_args(run_id=2, num_exp=2))
    registry = {
        "large": {
            "args": baseline_key_from_args(_args(run_id=0, num_exp=10))[1],
            "best_psnr_list": list(range(10)),
        },
        "small": {
            "args": baseline_key_from_args(_args(run_id=1, num_exp=4))[1],
            "best_psnr_list": list(range(4)),
        },
    }

    stored_key, _ = find_registry_entry(registry, key, args)

    assert stored_key == "small"


def test_legacy_lookup_rejects_single_entry_that_does_not_cover_requested_range():
    key, args = baseline_key_from_args(_args(run_id=0, num_exp=5))
    registry = {
        "short": {
            "args": baseline_key_from_args(_args(run_id=0, num_exp=3))[1],
            "best_psnr_list": [1.0, 2.0, 3.0],
        },
    }

    stored_key, entry = find_registry_entry(registry, key, args)

    assert stored_key is None
    assert entry is None


def test_registry_replaces_read_only_legacy_range_in_v2_with_new_ssim():
    writable_registry = {}
    key, args = baseline_key_from_args(_args(run_id=0, num_exp=3))
    legacy_entry = {
        "args": args,
        "best_psnr_list": [1.0, 2.0, 3.0],
        "best_mse_list": [4.0, 5.0, 6.0],
    }
    legacy_registry = {"legacy-key": legacy_entry}

    seed_registry_entry_from_fallback(writable_registry, legacy_registry, key, args)
    entry = update_idlg_baseline(writable_registry, key, args, [10, 11, 12], [13, 14, 15], [16, 17, 18])

    assert legacy_registry == {"legacy-key": legacy_entry}
    assert entry["best_psnr_list"] == [10.0, 11.0, 12.0]
    assert entry["best_mse_list"] == [13.0, 14.0, 15.0]
    assert entry["best_ssim_list"] == [16.0, 17.0, 18.0]


def test_registry_keeps_missing_historical_ssim_sparse_before_append():
    registry = {}
    key, first_args = baseline_key_from_args(_args(run_id=0, num_exp=3))
    _, second_args = baseline_key_from_args(_args(run_id=3, num_exp=2))
    registry[key] = {
        "args": first_args,
        "best_psnr_list": [1.0, 2.0, 3.0],
        "best_mse_list": [4.0, 5.0, 6.0],
    }

    entry = update_idlg_baseline(registry, key, second_args, [10, 11], [12, 13], [14, 15])

    assert "best_ssim_list" not in entry
    assert entry["best_ssim_by_run_id"] == {"3": 14.0, "4": 15.0}


def test_masked_only_comparison_uses_run_id_offset():
    baseline_entry = {
        "args": {"run_id": 0, "num_exp": 4},
        "best_psnr_list": [10.0, 20.0, 30.0, 40.0],
        "best_mse_list": [0.4, 0.3, 0.2, 0.1],
        "best_ssim_list": [0.1, 0.2, 0.3, 0.4],
    }
    results = {
        0: {"best_psnr_masked": 31.0, "best_mse_iDLG_masked": 0.19, "best_ssim_masked": 0.31},
        1: {"best_psnr_masked": 42.0, "best_mse_iDLG_masked": 0.08, "best_ssim_masked": 0.42},
    }

    report = paired_report_for_masked(results, baseline_entry, run_id=2)

    assert report["psnr_paired_stats"]["mean_diff"] == pytest.approx(1.5)
    assert report["mse_paired_stats"]["mean_diff"] == pytest.approx(-0.015)
    assert report["ssim_paired_stats"]["mean_diff"] == pytest.approx(0.015)


def test_masked_only_comparison_skips_missing_historical_ssim():
    baseline_entry = {
        "args": {"run_id": 0, "num_exp": 3},
        "best_psnr_list": [10.0, 20.0, 30.0],
        "best_mse_list": [0.4, 0.3, 0.2],
    }
    results = {
        0: {"best_psnr_masked": 21.0, "best_mse_iDLG_masked": 0.29, "best_ssim_masked": 0.21},
        1: {"best_psnr_masked": 32.0, "best_mse_iDLG_masked": 0.18, "best_ssim_masked": 0.32},
    }

    report = paired_report_for_masked(results, baseline_entry, run_id=1)

    assert report["psnr_paired_stats"]["mean_diff"] == pytest.approx(1.5)
    assert report["mse_paired_stats"]["mean_diff"] == pytest.approx(-0.015)
    assert report["ssim_ci_str"] == ""


def test_masked_only_comparison_uses_sparse_future_ssim():
    baseline_entry = {
        "args": {"run_id": 0, "num_exp": 4},
        "best_psnr_list": [10.0, 20.0, 30.0, 40.0],
        "best_mse_list": [0.4, 0.3, 0.2, 0.1],
        "best_ssim_by_run_id": {"2": 0.3, "3": 0.4},
    }
    results = {
        0: {"best_psnr_masked": 31.0, "best_mse_iDLG_masked": 0.19, "best_ssim_masked": 0.31},
        1: {"best_psnr_masked": 42.0, "best_mse_iDLG_masked": 0.08, "best_ssim_masked": 0.42},
    }

    report = paired_report_for_masked(results, baseline_entry, run_id=2)

    assert report["ssim_paired_stats"]["mean_diff"] == pytest.approx(0.015)


def test_masked_registry_lists_are_ordered_by_experiment_index():
    results = {
        2: {"best_psnr_masked": 30.0, "best_mse_iDLG_masked": 0.3, "best_ssim_masked": 0.33},
        0: {"best_psnr_masked": 10.0, "best_mse_iDLG_masked": 0.1, "best_ssim_masked": 0.11},
        1: {"best_psnr_masked": 20.0, "best_mse_iDLG_masked": 0.2, "best_ssim_masked": 0.22},
    }

    psnr, mse, ssim = ordered_masked_registry_lists(results)

    assert psnr == [10.0, 20.0, 30.0]
    assert mse == [0.1, 0.2, 0.3]
    assert ssim == [0.11, 0.22, 0.33]
