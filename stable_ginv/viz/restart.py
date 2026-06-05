"""Restart-curve CSV/PNG and representative per-restart image grid (Phase 7).

Extracted verbatim from helper/visualization.py: per-restart PSNR statistics,
paired restart-gain confidence intervals (vs k=1), the restart-curve CSV+PNG
writer, and the representative per-restart reconstruction image grid.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from stable_ginv.io import safe_savefig, safe_write
from stable_ginv.stats import normality_str_from_ci, paired_t_ci

sns.set_theme(style="whitegrid")


def _restart_stats(per_exp_lists):
    arr = np.array(
        [[v if v is not None else float("nan") for v in row] for row in per_exp_lists],
        dtype=float,
    )
    n = np.sum(~np.isnan(arr), axis=0).clip(min=1)
    means = np.nanmean(arr, axis=0)
    stds = np.nanstd(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros(arr.shape[1])
    sems = stds / np.sqrt(n)
    return means, stds, sems


def _gain_ci_str(all_rows, k, fmt=".3f"):
    pairs = [
        (row[0], row[k - 1]) for row in (all_rows or [])
        if row and row[0] is not None and row[k - 1] is not None
    ]
    if len(pairs) < 2:
        return "n/a", None
    x_k = [p[1] for p in pairs]
    x_1 = [p[0] for p in pairs]
    ci = paired_t_ci(x_k, x_1)
    sign = "+" if ci["mean_diff"] >= 0 else ""
    s = (
        f"{sign}{ci['mean_diff']:{fmt}} "
        f"[{ci['ci_low']:{fmt}}, {ci['ci_high']:{fmt}}]"
        f" p={ci['p_value']:.3f}"
    )
    return s, ci


def save_restart_curve(save_dir, timestamp_str, num_restarts, network_name, dataset,
                       mask_mode, psnr_per_restart_idlg_all,
                       psnr_per_restart_masked_all, mse_per_restart_idlg_all,
                       mse_per_restart_masked_all):
    """Save restart-curve CSV/PNG and print paired gain summaries."""
    restart_csv_path = os.path.join(save_dir, f"restart_curve_{timestamp_str}.csv")
    restart_x = list(range(1, num_restarts + 1))

    # Restart statistics do not depend on k, so compute them once per method
    # rather than recomputing the full arrays inside the restart-count loop.
    means_i = stds_i = means_m = stds_m = None
    if psnr_per_restart_idlg_all:
        means_i, stds_i, _ = _restart_stats(psnr_per_restart_idlg_all)
    if psnr_per_restart_masked_all:
        means_m, stds_m, _ = _restart_stats(psnr_per_restart_masked_all)

    rows_restart = []
    for k in range(num_restarts):
        row = {"num_restarts": k + 1}
        if psnr_per_restart_idlg_all:
            row["mean_psnr_idlg"] = round(float(means_i[k]), 5)
            row["std_psnr_idlg"] = round(float(stds_i[k]), 5)
        if psnr_per_restart_masked_all:
            row["mean_psnr_masked"] = round(float(means_m[k]), 5)
            row["std_psnr_masked"] = round(float(stds_m[k]), 5)
        rows_restart.append(row)

    restart_fieldnames = ["num_restarts"]
    if psnr_per_restart_idlg_all:
        restart_fieldnames += ["mean_psnr_idlg", "std_psnr_idlg"]
    if psnr_per_restart_masked_all:
        restart_fieldnames += ["mean_psnr_masked", "std_psnr_masked"]

    def _write_restart_csv(f):
        writer = csv.DictWriter(f, fieldnames=restart_fieldnames)
        writer.writeheader()
        writer.writerows(rows_restart)

    safe_write(restart_csv_path, _write_restart_csv, newline="")

    line_rows = []
    for method_label, per_exp_lists in [
        ("iDLG (no mask)", psnr_per_restart_idlg_all),
        (f"masked ({mask_mode})", psnr_per_restart_masked_all),
    ]:
        for row in per_exp_lists or []:
            for restart_count, value in zip(restart_x, row):
                if value is not None and np.isfinite(value):
                    line_rows.append({
                        "Number of restarts used": restart_count,
                        "Best PSNR (dB)": float(value),
                        "Method": method_label,
                    })

    fig, ax = plt.subplots(figsize=(6, 4))
    if line_rows:
        sns.lineplot(
            data=pd.DataFrame(line_rows),
            x="Number of restarts used",
            y="Best PSNR (dB)",
            hue="Method",
            style="Method",
            markers=True,
            dashes=False,
            errorbar="se",
            ax=ax,
        )
    ax.set_xlabel("Number of restarts used")
    ax.set_ylabel("Mean best PSNR (dB) ± SEM")
    ax.set_title(f"Effect of restarts — {network_name} / {dataset}")
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.12)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    restart_plot_path = os.path.join(save_dir, f"restart_curve_{timestamp_str}.png")
    safe_savefig(fig, restart_plot_path, dpi=200)
    plt.close(fig)
    print(f"\nRestart curve saved: {restart_csv_path}, {restart_plot_path}")

    gain_ks = [k for k in [3, 5, 10] if k <= num_restarts]
    gain_rows_extra = {k: {} for k in gain_ks}

    if gain_ks:
        print("\nRestart gain summary vs k=1 (95% paired CI):")
        header = f"  {'':10s}"
        for k in gain_ks:
            header += f"  PSNR gain k={k:<3d}              MSE gain k={k:<3d}    "
        print(header)

        for method_label, psnr_all, mse_all in [
            ("iDLG", psnr_per_restart_idlg_all, mse_per_restart_idlg_all),
            ("masked", psnr_per_restart_masked_all, mse_per_restart_masked_all),
        ]:
            if not psnr_all and not mse_all:
                continue
            gain_key = "idlg" if method_label == "iDLG" else "masked"
            row_str = f"  {method_label:<10s}"
            norm_str = f"  {'':10s}"
            for k in gain_ks:
                psnr_str, psnr_ci = _gain_ci_str(psnr_all, k, ".3f")
                mse_str, mse_ci = _gain_ci_str(mse_all, k, ".5f")
                row_str += f"  {psnr_str:<32s}  {mse_str:<32s}"
                psnr_norm = normality_str_from_ci(psnr_ci) if psnr_ci else "normality: n/a"
                mse_norm = normality_str_from_ci(mse_ci) if mse_ci else "normality: n/a"
                norm_str += f"  {psnr_norm:<44s}  {mse_norm:<44s}"
                if psnr_ci is not None:
                    gain_rows_extra[k][f"gain_psnr_{gain_key}"] = round(psnr_ci["mean_diff"], 5)
                    gain_rows_extra[k][f"ci_low_psnr_{gain_key}"] = round(psnr_ci["ci_low"], 5)
                    gain_rows_extra[k][f"ci_high_psnr_{gain_key}"] = round(psnr_ci["ci_high"], 5)
                    gain_rows_extra[k][f"normality_psnr_{gain_key}"] = normality_str_from_ci(psnr_ci)
                if mse_ci is not None:
                    gain_rows_extra[k][f"gain_mse_{gain_key}"] = round(mse_ci["mean_diff"], 7)
                    gain_rows_extra[k][f"ci_low_mse_{gain_key}"] = round(mse_ci["ci_low"], 7)
                    gain_rows_extra[k][f"ci_high_mse_{gain_key}"] = round(mse_ci["ci_high"], 7)
                    gain_rows_extra[k][f"normality_mse_{gain_key}"] = normality_str_from_ci(mse_ci)
            print(row_str)
            print(norm_str)

    if gain_ks and any(gain_rows_extra[k] for k in gain_ks):
        gain_extra_fields = []
        for k in gain_ks:
            gain_extra_fields += list(gain_rows_extra[k].keys())
        gain_extra_fields = list(dict.fromkeys(gain_extra_fields))
        new_fieldnames = restart_fieldnames + [
            f for f in gain_extra_fields if f not in restart_fieldnames
        ]
        for row in rows_restart:
            if row["num_restarts"] in gain_rows_extra:
                row.update(gain_rows_extra[row["num_restarts"]])

        def _rewrite_restart_csv(f):
            writer = csv.DictWriter(f, fieldnames=new_fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows_restart)

        safe_write(restart_csv_path, _rewrite_restart_csv, newline="")

    return restart_csv_path, restart_plot_path


def save_restart_images(save_dir, timestamp_str, num_restarts, network_name, dataset,
                        mask_mode, num_exp, all_results_by_idx,
                        psnr_per_restart_idlg_all, psnr_per_restart_masked_all):
    """Save representative per-restart reconstruction image grid."""
    if not all_results_by_idx:
        return None
    display_ks = [k for k in [1, 3, 5, 10] if k <= num_restarts]
    if not display_ks:
        return None

    ref_k = max(display_ks)
    ref_pool = psnr_per_restart_idlg_all or psnr_per_restart_masked_all
    rep_result = None
    if num_exp > 1 and ref_pool:
        ref_vals = [row[ref_k - 1] for row in ref_pool if row and row[ref_k - 1] is not None]
        if ref_vals:
            median_val = float(np.median(ref_vals))
            sorted_results = sorted(
                all_results_by_idx.values(),
                key=lambda r: abs(
                    (
                        (r.get("psnr_per_restart_idlg") or
                         r.get("psnr_per_restart_masked") or [None])[ref_k - 1] or float("inf")
                    ) - median_val
                ),
            )
            rep_result = sorted_results[0]
    if rep_result is None:
        rep_result = next(iter(all_results_by_idx.values()))

    imgs_i = rep_result.get("img_per_restart_idlg") or []
    imgs_m = rep_result.get("img_per_restart_masked") or []
    psnrs_i = rep_result.get("psnr_per_restart_idlg") or []
    psnrs_m = rep_result.get("psnr_per_restart_masked") or []
    mses_i = rep_result.get("mse_per_restart_idlg") or []
    mses_m = rep_result.get("mse_per_restart_masked") or []
    ssims_i = rep_result.get("ssim_per_restart_idlg") or []
    ssims_m = rep_result.get("ssim_per_restart_masked") or []
    gt_np = rep_result["gt_data"]

    def _to_hwc(arr):
        if arr is None:
            return None
        x = arr[0]
        return x.transpose(1, 2, 0) if x.shape[0] > 1 else x[0]

    def _metric_str(k, mse_list, psnr_list, ssim_list):
        c = display_ks.index(k)
        parts = []
        m = mse_list[c] if c < len(mse_list) else None
        p = psnr_list[c] if c < len(psnr_list) else None
        s = ssim_list[c] if c < len(ssim_list) else None
        if m is not None and np.isfinite(m):
            parts.append(f"MSE: {m:.5f}")
        if p is not None and np.isfinite(p):
            parts.append(f"PSNR: {p:.2f} dB")
        if s is not None and np.isfinite(s):
            parts.append(f"SSIM: {s:.3f}")
        return "\n".join(parts)

    def _disp_list(src, is_gt=False):
        if is_gt:
            return [gt_np] * len(display_ks)
        return [src[k - 1] if (k - 1) < len(src) else None for k in display_ks]

    rows_fig = [("GT", _disp_list([], is_gt=True), [], [], [])]
    if imgs_i:
        rows_fig.append(("iDLG\n(baseline)", _disp_list(imgs_i),
                         _disp_list(psnrs_i), _disp_list(mses_i), _disp_list(ssims_i)))
    if imgs_m:
        rows_fig.append((f"masked\n({mask_mode})", _disp_list(imgs_m),
                         _disp_list(psnrs_m), _disp_list(mses_m), _disp_list(ssims_m)))

    n_cols_fig = len(display_ks)
    n_rows_fig = len(rows_fig)
    fig_img, axes = plt.subplots(
        n_rows_fig, n_cols_fig,
        figsize=(3.2 * n_cols_fig + 0.8, 3.8 * n_rows_fig),
        squeeze=False,
    )
    fig_img.subplots_adjust(left=0.12, right=0.98, top=0.93, bottom=0.02,
                            hspace=0.35, wspace=0.05)
    cmap = "gray" if gt_np.shape[1] == 1 else None
    for r_idx, (label, img_list, psnr_list, mse_list, ssim_list) in enumerate(rows_fig):
        for c_idx in range(n_cols_fig):
            k = display_ks[c_idx]
            ax = axes[r_idx][c_idx]
            hwc = _to_hwc(img_list[c_idx])
            if hwc is not None:
                ax.imshow(hwc.clip(0, 1), cmap=cmap)
            else:
                ax.set_facecolor("lightgray")
            ax.axis("off")
            if r_idx == 0:
                ax.set_title(f"k={k}", fontsize=10)
            if r_idx > 0 and psnr_list:
                ms = _metric_str(k, mse_list, psnr_list, ssim_list)
                if ms:
                    ax.text(0.5, -0.02, ms, transform=ax.transAxes,
                            ha="center", va="top", fontsize=7, linespacing=1.5)
        axes[r_idx][0].text(-0.08, 0.5, label, transform=axes[r_idx][0].transAxes,
                            ha="right", va="center", fontsize=9,
                            rotation=90, multialignment="center")

    title_suffix = f" (representative of {num_exp} images)" if num_exp > 1 else ""
    fig_img.suptitle(
        f"Best reconstruction after k restarts — {network_name} / {dataset}{title_suffix}",
        fontsize=10,
    )
    plt.tight_layout()
    img_fig_path = os.path.join(save_dir, f"restart_images_{timestamp_str}.png")
    safe_savefig(fig_img, img_fig_path, dpi=200)
    plt.close(fig_img)
    print(f"Restart image figure saved: {img_fig_path}")
    return img_fig_path
