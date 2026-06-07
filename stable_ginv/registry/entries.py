"""Find, seed, and update registry entries for contiguous sample ranges."""
import copy

import numpy as np


def _config_without_sample_range(args):
    return {
        name: value
        for name, value in args.items()
        if name not in ("num_exp", "run_id")
    }


def _entry_sample_range(entry):
    args = entry.get("args", {})
    if "run_id" not in args:
        return None
    start = int(args["run_id"])
    count = len(entry.get("best_psnr_list", []))
    return start, start + count


def _requested_sample_range(comparable_args):
    if "run_id" not in comparable_args or "num_exp" not in comparable_args:
        return None
    start = int(comparable_args["run_id"])
    return start, start + int(comparable_args["num_exp"])


def find_registry_entry(registry, key, comparable_args, allow_append_predecessor=False):
    """Return the best current or legacy registry entry matching one configuration."""
    target_config = _config_without_sample_range(comparable_args)
    matches = [
        (stored_key, entry)
        for stored_key, entry in registry.items()
        if _config_without_sample_range(entry.get("args", {})) == target_config
    ]
    requested_range = _requested_sample_range(comparable_args)
    if requested_range is None:
        if key in registry:
            return key, registry[key]
        if len(matches) <= 1:
            return matches[0] if matches else (None, None)
        raise ValueError(
            "Multiple legacy registry entries match this configuration. "
            "Provide a sample range or select the registry entry explicitly."
        )

    requested_start, requested_end = requested_range
    containing = [
        (stored_key, entry, start, end)
        for stored_key, entry in matches
        if (sample_range := _entry_sample_range(entry)) is not None
        for start, end in [sample_range]
        if start <= requested_start and requested_end <= end
    ]
    if containing:
        stored_key, entry, _, _ = min(
            containing,
            key=lambda item: (
                item[3] - item[2],
                item[2],
                item[0],
            ),
        )
        return stored_key, entry

    if allow_append_predecessor:
        adjacent = [
            (stored_key, entry, start, end)
            for stored_key, entry in matches
            if (sample_range := _entry_sample_range(entry)) is not None
            for start, end in [sample_range]
            if end == requested_start or start == requested_end
        ]
        if adjacent:
            stored_key, entry, _, _ = max(
                adjacent,
                key=lambda item: (
                    item[3] - item[2],
                    -item[2],
                    item[0],
                ),
            )
            return stored_key, entry

    return None, None


def seed_registry_entry_from_fallback(registry, fallback_registry, key, comparable_args):
    """Copy one matching read-only fallback entry into a writable registry."""
    if key in registry or find_registry_entry(registry, key, comparable_args)[1] is not None:
        return
    _, entry = find_registry_entry(
        fallback_registry, key, comparable_args, allow_append_predecessor=True
    )
    if entry is not None:
        registry[key] = copy.deepcopy(entry)


def _replace_ssim_range(entry, stored_start, stored_count, incoming_start,
                        incoming_count, incoming_ssim):
    """Replace SSIM values for one stored range while preserving sparse legacy gaps."""
    replace_ids = {str(incoming_start + offset) for offset in range(incoming_count)}
    sparse_ssim = {
        str(stored_start + offset): float(value)
        for offset, value in enumerate(entry.get("best_ssim_list", []))
    }
    sparse_ssim.update({
        str(run_id): float(value)
        for run_id, value in entry.get("best_ssim_by_run_id", {}).items()
    })
    for run_id in replace_ids:
        sparse_ssim.pop(run_id, None)
    sparse_ssim.update({
        str(incoming_start + offset): value
        for offset, value in enumerate(incoming_ssim)
    })

    dense_ssim = [
        sparse_ssim.get(str(stored_start + offset))
        for offset in range(stored_count)
    ]
    if all(value is not None for value in dense_ssim):
        entry["best_ssim_list"] = dense_ssim
        entry.pop("best_ssim_by_run_id", None)
    else:
        entry.pop("best_ssim_list", None)
        entry["best_ssim_by_run_id"] = sparse_ssim


def _update_registry_entry(registry, key, comparable_args, best_psnr_list,
                           best_mse_list, best_ssim_list, label):
    """Store metrics for one sample range, merging by absolute run_id."""
    incoming = {
        "best_psnr_list": [float(v) for v in best_psnr_list],
        "best_mse_list": [float(v) for v in best_mse_list],
    }
    lengths = {len(values) for values in incoming.values()}
    if len(lengths) != 1:
        raise ValueError(f"{label} PSNR and MSE lists must have equal lengths.")

    incoming_start = int(comparable_args["run_id"])
    incoming_count = lengths.pop()
    if incoming_count != int(comparable_args["num_exp"]):
        raise ValueError(
            f"{label} received {incoming_count} metric values for "
            f"num_exp={comparable_args['num_exp']}."
        )
    incoming["best_ssim_list"] = [float(v) for v in best_ssim_list]
    if len(incoming["best_ssim_list"]) not in (0, incoming_count):
        raise ValueError(f"{label} SSIM list must be empty or match the PSNR and MSE lists.")

    if key in registry:
        stored_key, entry = key, registry[key]
    else:
        stored_key, entry = find_registry_entry(
            registry, key, comparable_args, allow_append_predecessor=True
        )
    if entry is None:
        registry[key] = {
            "args": dict(comparable_args),
            **incoming,
        }
        return registry[key]

    if stored_key != key:
        entry = copy.deepcopy(entry)
        registry[key] = entry
        print(f"\nCopied legacy {label} registry entry to appendable key {key}.")

    entry_args = entry["args"]
    stored_start = int(entry_args["run_id"])
    stored_count = len(entry["best_psnr_list"])
    stored_ssim_list = entry.get("best_ssim_list", [])
    if len(stored_ssim_list) > stored_count:
        raise ValueError(f"Stored {label} SSIM list is longer than the PSNR list.")
    stored_end = stored_start + stored_count
    incoming_end = incoming_start + incoming_count

    merged_start = min(stored_start, incoming_start)
    merged_end = max(stored_end, incoming_end)
    stored_ids = set(range(stored_start, stored_end))
    incoming_ids = set(range(incoming_start, incoming_end))
    replace_existing = stored_start <= incoming_start and incoming_end <= stored_end
    incoming_update_ids = incoming_ids if replace_existing else incoming_ids - stored_ids
    merged_ids = stored_ids | incoming_ids
    expected_ids = set(range(merged_start, merged_end))
    if merged_ids != expected_ids:
        missing = sorted(expected_ids - merged_ids)
        raise ValueError(
            f"Cannot merge {label} run_id={incoming_start}..{incoming_end - 1}: "
            f"stored samples cover run_id={stored_start}..{stored_end - 1}, "
            f"leaving missing run_id={missing[0]}."
        )

    psnr_by_run_id = {
        stored_start + offset: float(value)
        for offset, value in enumerate(entry["best_psnr_list"])
    }
    mse_by_run_id = {
        stored_start + offset: float(value)
        for offset, value in enumerate(entry["best_mse_list"])
    }
    psnr_by_run_id.update({
        incoming_start + offset: value
        for offset, value in enumerate(incoming["best_psnr_list"])
        if incoming_start + offset in incoming_update_ids
    })
    mse_by_run_id.update({
        incoming_start + offset: value
        for offset, value in enumerate(incoming["best_mse_list"])
        if incoming_start + offset in incoming_update_ids
    })

    ssim_by_run_id = {
        stored_start + offset: float(value)
        for offset, value in enumerate(entry.get("best_ssim_list", []))
    }
    ssim_by_run_id.update({
        int(run_id): float(value)
        for run_id, value in entry.get("best_ssim_by_run_id", {}).items()
    })
    for run_id in incoming_update_ids:
        ssim_by_run_id.pop(run_id, None)
    ssim_by_run_id.update({
        incoming_start + offset: value
        for offset, value in enumerate(incoming["best_ssim_list"])
        if incoming_start + offset in incoming_update_ids
    })

    ordered_ids = list(range(merged_start, merged_end))
    entry["best_psnr_list"] = [psnr_by_run_id[run_id] for run_id in ordered_ids]
    entry["best_mse_list"] = [mse_by_run_id[run_id] for run_id in ordered_ids]
    ordered_ssim = [ssim_by_run_id.get(run_id) for run_id in ordered_ids]
    if all(value is not None for value in ordered_ssim):
        entry["best_ssim_list"] = ordered_ssim
        entry.pop("best_ssim_by_run_id", None)
    elif any(value is not None for value in ordered_ssim):
        entry.pop("best_ssim_list", None)
        entry["best_ssim_by_run_id"] = {
            str(run_id): ssim_by_run_id[run_id]
            for run_id in ordered_ids
            if run_id in ssim_by_run_id
        }
    else:
        entry.pop("best_ssim_list", None)
        entry.pop("best_ssim_by_run_id", None)

    entry_args["run_id"] = merged_start
    entry_args["num_exp"] = len(ordered_ids)
    overlap_count = len(stored_ids & incoming_ids)
    replaced_count = overlap_count if replace_existing else 0
    skipped_count = overlap_count - replaced_count
    new_count = len(incoming_update_ids) - replaced_count
    print(
        f"\nMerged {label} run_id={incoming_start}..{incoming_end - 1}; "
        f"added {new_count} new sample(s), replaced {replaced_count} existing sample(s), "
        f"skipped {skipped_count} existing sample(s), "
        f"registry now covers run_id={merged_start}..{merged_end - 1}."
    )
    return entry


def update_masked_registry(registry, key, comparable_args, best_psnr_list, best_mse_list, best_ssim_list):
    """Store or append a contiguous masked sample range for one configuration."""
    return _update_registry_entry(
        registry, key, comparable_args, best_psnr_list, best_mse_list, best_ssim_list,
        label="masked",
    )


def update_idlg_baseline(registry, key, comparable_args, best_psnr_list, best_mse_list, best_ssim_list):
    """Store or append a contiguous iDLG baseline sample range for one configuration."""
    return _update_registry_entry(
        registry, key, comparable_args, best_psnr_list, best_mse_list, best_ssim_list,
        label="iDLG baseline",
    )


def available_ssim_values(entry):
    """Return finite SSIM values from legacy lists and sparse future samples."""
    values = [float(v) for v in entry.get("best_ssim_list", [])]
    values.extend(float(v) for v in entry.get("best_ssim_by_run_id", {}).values())
    return [value for value in values if np.isfinite(value)]
