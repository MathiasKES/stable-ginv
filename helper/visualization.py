import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import os

import numpy as np
import pandas as pd
import torch

from functions.io_utils import (
    normality_str_from_ci,
    paired_t_ci,
    safe_savefig,
    safe_chmod,
    safe_makedirs,
    safe_write,
)

sns.set_theme(style="whitegrid")


def create_panel_buffers():
    """Create named reconstruction-panel buffers."""
    return {
        "exp_idx": [],
        "gt": [],
        "idlg": [],
        "masked": [],
        "psnr_idlg": [],
        "ssim_idlg": [],
        "mse_idlg": [],
        "psnr_masked": [],
        "ssim_masked": [],
        "mse_masked": [],
    }


def append_result_to_panel_buffers(result, buffers, to_pil, warn_fn):
    """Append one experiment result to panel buffers."""
    idx = result["idx_net"]
    buffers["exp_idx"].append(idx)
    gt_pil = to_pil(torch.from_numpy(result["gt_data"])[0])
    buffers["gt"].append(gt_pil)

    if "iDLG" not in result["final_recon"]:
        idlg_pil = gt_pil
    elif result.get("best_psnr_idlg") is None:
        warn_fn(
            f"[WARNING] Experiment {idx}: iDLG reconstruction failed "
            f"(all restarts diverged); showing blank in panel."
        )
        idlg_pil = gt_pil
    else:
        idlg_pil = to_pil(torch.from_numpy(result["final_recon"]["iDLG"])[0])
    buffers["idlg"].append(idlg_pil)

    if "iDLG_masked" not in result["final_recon"]:
        masked_pil = gt_pil
    elif result.get("best_psnr_masked") is None:
        warn_fn(
            f"[WARNING] Experiment {idx}: masked iDLG reconstruction failed "
            f"(all restarts diverged); showing blank in panel."
        )
        masked_pil = gt_pil
    else:
        masked_pil = to_pil(torch.from_numpy(result["final_recon"]["iDLG_masked"])[0])
    buffers["masked"].append(masked_pil)

    buffers["psnr_idlg"].append(result.get("best_psnr_idlg"))
    buffers["ssim_idlg"].append(result.get("best_ssim_idlg"))
    buffers["mse_idlg"].append(result.get("best_mse_iDLG"))
    buffers["psnr_masked"].append(result.get("best_psnr_masked"))
    buffers["ssim_masked"].append(result.get("best_ssim_masked"))
    buffers["mse_masked"].append(result.get("best_mse_iDLG_masked"))


def flush_recon_panel(params, buffers, panel_png_paths, save_dir, block_idx,
                      dataset, mask_desc, timestamp_str, methods):
    """Save current reconstruction panel buffers and clear them."""
    if len(buffers["gt"]) == 0:
        return block_idx
    order = sorted(range(len(buffers["exp_idx"])), key=lambda i: buffers["exp_idx"][i])
    ordered = {
        name: [values[i] for i in order]
        for name, values in buffers.items()
    }
    panel_path = save_recon_panel(
        params, ordered["gt"], ordered["idlg"], ordered["masked"],
        save_dir, block_idx, dataset, mask_desc, timestamp_str,
        methods=methods,
        exp_indices=ordered["exp_idx"],
        psnr_idlg=ordered["psnr_idlg"],
        ssim_idlg=ordered["ssim_idlg"],
        mse_idlg=ordered["mse_idlg"],
        psnr_masked=ordered["psnr_masked"],
        ssim_masked=ordered["ssim_masked"],
        mse_masked=ordered["mse_masked"],
    )
    if panel_path:
        panel_png_paths.append(panel_path)
    for values in buffers.values():
        values.clear()
    return block_idx + 1


def save_recon_panel(params: dict, panel_gt_pil, panel_idlg_pil, panel_masked_pil,
                     save_dir, block_idx, dataset, mask_desc: str, timestamp_str: str,
                     methods: str = "both",
                     exp_indices=None,
                     psnr_idlg=None, ssim_idlg=None, mse_idlg=None,
                     psnr_masked=None, ssim_masked=None, mse_masked=None):
    """Save a PNG reconstruction panel; rows adapt to which methods were run."""
    n = len(panel_gt_pil)
    if n == 0:
        return

    rows = [("GT", panel_gt_pil, None, None, None)]

    if methods in ["idlg", "both"]:
        rows.append(("iDLG", panel_idlg_pil, psnr_idlg, ssim_idlg, mse_idlg))

    if methods in ["masked", "both"]:
        rows.append(("iDLG_masked", panel_masked_pil, psnr_masked, ssim_masked, mse_masked))

    num_rows = len(rows)
    fig = plt.figure(figsize=(2.2 * n + 1.6, 2.5 * num_rows))

    for r, (row_name, row_imgs, row_psnr, row_ssim, row_mse) in enumerate(rows):
        y = 1.0 - (r + 0.5) / num_rows
        fig.text(0.01, y, row_name, va='center', ha='left',
                 fontsize=14, fontweight='bold')

        for j in range(n):
            ax = plt.subplot(num_rows, n, r * n + 1 + j)
            ax.imshow(row_imgs[j], cmap='gray' if dataset == 'MNIST' else None)
            if r == 0:
                exp_idx = exp_indices[j] if exp_indices is not None else j
                ax.set_title(f"exp {exp_idx}", fontsize=8)

            ax.axis('off')
            if row_psnr is not None or row_ssim is not None or row_mse is not None:
                parts = []
                if row_psnr is not None and row_psnr[j] is not None:
                    parts.append(f"PSNR:{row_psnr[j]:.2f}dB")
                if row_ssim is not None and row_ssim[j] is not None:
                    parts.append(f"SSIM:{row_ssim[j]:.3f}")
                if row_mse is not None and row_mse[j] is not None:
                    parts.append(f"MSE:{row_mse[j]:.4f}")
                ax.text(0.5, -0.05, "\n".join(parts), transform=ax.transAxes,
                        fontsize=7, ha='center', va='top')

    plt.tight_layout(rect=(0.08, 0.0, 1.0, 1.0))
    if os.environ.get("LSB_INTERACTIVE") == "Y":
        job_id = "INTERACTIVE"
    else:
        job_id = os.environ.get("LSB_JOBID", "")
    prefix = f"{timestamp_str}_{job_id}" if job_id else timestamp_str
    out_path = os.path.join(save_dir, f"{prefix}_{block_idx}.png")
    ok = safe_savefig(fig, out_path, dpi=75, bbox_inches='tight')
    plt.close(fig)

    if not ok:
        return None
    print("Saved reconstruction panel to:", out_path)
    return out_path


def save_recon_gif(
    results_list,
    save_dir, block_idx, dataset, mask_desc, timestamp_str,
    methods="both", fps=8,
):
    """Save an animated GIF of reconstruction progress; rows=experiments, cols=[Init|iDLG?|Masked?|GT]."""
    import io
    from PIL import Image as PILImage

    n_exp = len(results_list)
    if n_exp == 0:
        return None

    show_idlg = methods in ["idlg", "both"]
    show_masked = methods in ["masked", "both"]

    def _to_pil(img_np):
        arr = np.clip(img_np, 0.0, 1.0)
        if arr.shape[0] == 1:
            return PILImage.fromarray((arr[0] * 255).astype(np.uint8), mode='L')
        return PILImage.fromarray(
            (np.transpose(arr, (1, 2, 0)) * 255).astype(np.uint8), mode='RGB'
        )

    max_frames = 1
    for r in results_list:
        for mk in (["iDLG"] if show_idlg else []) + (["iDLG_masked"] if show_masked else []):
            n = len(r.get("recon_frames", {}).get(mk, []))
            if n > max_frames:
                max_frames = n

    cols = ["init"]
    if show_idlg:
        cols.append("iDLG")
    if show_masked:
        cols.append("masked")
    cols.append("GT")
    n_cols = len(cols)

    both = show_idlg and show_masked
    col_title = {
        "init": "Random Init",
        "iDLG": "iDLG" if both else "Recovered",
        "masked": "Masked" if both else "Recovered",
        "GT": "Ground Truth",
    }

    total_frames = max_frames + 5

    gif_frames = []
    for t in range(total_frames):
        frame_t = min(t, max_frames - 1)

        fig_w = max(4.0, 1.8 * n_cols + 0.6)
        fig_h = max(3.0, 1.8 * n_exp + 0.4)
        fig, axes = plt.subplots(
            n_exp, n_cols, figsize=(fig_w, fig_h),
            squeeze=False,
            gridspec_kw={'hspace': 0.45, 'wspace': 0.05},
        )

        frame_iter = 0
        for r in results_list:
            for mk in ["iDLG", "iDLG_masked"]:
                frs = r.get("recon_frames", {}).get(mk, [])
                if frs:
                    frame_iter = frs[min(frame_t, len(frs) - 1)].get('iter', 0)
                    break
            else:
                continue
            break
        fig.suptitle(f"Iteration {frame_iter}", fontsize=9)

        cmap = 'gray' if dataset == 'MNIST' else None

        for row_idx, result in enumerate(results_list):
            y_pos = 1.0 - (row_idx + 0.5) / n_exp
            fig.text(0.005, y_pos, f"Exp {row_idx}", va='center', ha='left',
                     fontsize=6, fontweight='bold')

            for col_idx, col in enumerate(cols):
                ax = axes[row_idx, col_idx]
                ax.axis('off')

                if col == "GT":
                    ax.imshow(_to_pil(result['gt_data'][0]), cmap=cmap,
                              interpolation='nearest', aspect='equal')
                    if row_idx == 0:
                        ax.set_title(col_title["GT"], fontsize=7, pad=2)

                elif col == "init":
                    init_np = None
                    for mk in ["iDLG", "iDLG_masked"]:
                        init_np = result.get("init_frames", {}).get(mk)
                        if init_np is not None:
                            break
                    if init_np is not None:
                        ax.imshow(_to_pil(init_np[0]), cmap=cmap,
                                  interpolation='nearest', aspect='equal')
                    else:
                        ax.set_facecolor('#aaaaaa')
                    if row_idx == 0:
                        ax.set_title(col_title["init"], fontsize=7, pad=2)

                else:
                    mk = "iDLG" if col == "iDLG" else "iDLG_masked"
                    frs = result.get("recon_frames", {}).get(mk, [])
                    if frs:
                        fd = frs[min(frame_t, len(frs) - 1)]
                        ax.imshow(_to_pil(fd['dummy'][0]), cmap=cmap,
                                  interpolation='nearest', aspect='equal')
                        loss_v = fd.get('loss', float('inf'))
                        mse_v = fd.get('mse', float('inf'))
                        loss_str = f"{loss_v:.2e}" if np.isfinite(loss_v) else "inf"
                        ax.text(
                            0.5, -0.06,
                            f"GLoss:{loss_str}\nL2Loss:{mse_v:.4f}",
                            transform=ax.transAxes,
                            fontsize=5, va='top', ha='center',
                            clip_on=False,
                        )
                    else:
                        ax.set_facecolor('#aaaaaa')
                    if row_idx == 0:
                        ax.set_title(col_title.get(col, col), fontsize=7, pad=2)

        plt.tight_layout(rect=(0.04, 0.0, 1.0, 0.96))

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=125, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        gif_frames.append(PILImage.open(buf).copy())
        buf.close()

    if not gif_frames:
        return None

    job_id = "INTERACTIVE" if os.environ.get("LSB_INTERACTIVE") == "Y" else os.environ.get("LSB_JOBID", "")
    prefix = f"{timestamp_str}_{job_id}" if job_id else timestamp_str
    out_path = os.path.join(save_dir, f"{prefix}_{block_idx}_anim.gif")

    duration_ms = max(1, int(1000 / fps))
    parent = os.path.dirname(out_path)
    if parent and not safe_makedirs(parent):
        return None
    try:
        gif_frames[0].save(
            out_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
        )
    except OSError as e:
        print(f"[WARNING] Failed to save GIF {out_path}: {e}")
        return None

    safe_chmod(out_path)
    print("Saved animated GIF to:", out_path)
    return out_path


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
