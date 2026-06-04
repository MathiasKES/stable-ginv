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
        predecessors = [
            (stored_key, entry, start, end)
            for stored_key, entry in matches
            if (sample_range := _entry_sample_range(entry)) is not None
            for start, end in [sample_range]
            if end == requested_start
        ]
        if predecessors:
            stored_key, entry, _, _ = max(
                predecessors,
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
    """Store metrics for a sample range, appending unseen suffixes or replacing contained ranges."""
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
    expected_start = stored_start + stored_count
    incoming_end = incoming_start + incoming_count
    if stored_start <= incoming_start and incoming_end <= expected_start:
        offset = incoming_start - stored_start
        entry["best_psnr_list"][offset:offset + incoming_count] = incoming["best_psnr_list"]
        entry["best_mse_list"][offset:offset + incoming_count] = incoming["best_mse_list"]
        _replace_ssim_range(
            entry, stored_start, stored_count, incoming_start, incoming_count,
            incoming["best_ssim_list"],
        )
        print(
            f"\nReplaced {incoming_count} existing {label} sample(s) for "
            f"run_id={incoming_start}..{incoming_end - 1}."
        )
        return entry
    if stored_start <= incoming_start < expected_start < incoming_end:
        overlap_count = expected_start - incoming_start
        incoming_start = expected_start
        incoming_count -= overlap_count
        incoming = {
            name: values[overlap_count:]
            for name, values in incoming.items()
        }
        print(
            f"\nSkipped {overlap_count} existing {label} sample(s) and will append "
            f"run_id={incoming_start}..{incoming_end - 1}."
        )
    if incoming_start != expected_start:
        raise ValueError(
            f"Cannot append {label} run_id={incoming_start}: stored samples cover "
            f"run_id={stored_start}..{expected_start - 1}, so the next run must use "
            f"--run_id {expected_start}."
        )

    entry["best_psnr_list"].extend(incoming["best_psnr_list"])
    entry["best_mse_list"].extend(incoming["best_mse_list"])
    if len(stored_ssim_list) == stored_count:
        entry.setdefault("best_ssim_list", []).extend(incoming["best_ssim_list"])
    elif incoming["best_ssim_list"]:
        sparse_ssim = entry.setdefault("best_ssim_by_run_id", {})
        sparse_ssim.update({
            str(incoming_start + offset): value
            for offset, value in enumerate(incoming["best_ssim_list"])
        })
    entry_args["num_exp"] = stored_count + incoming_count
    print(
        f"\nAppended {incoming_count} {label} sample(s); "
        f"registry entry now contains {entry_args['num_exp']} sample(s)."
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
