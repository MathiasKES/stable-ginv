import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

import numpy as np


def save_recon_panel(params: dict, panel_gt_pil, panel_idlg_pil, panel_masked_pil,
                     save_dir, block_idx, dataset, mask_desc: str, timestamp_str: str,
                     methods: str = "both",
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
                ax.set_title(f"exp {j}", fontsize=8)

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
    plt.savefig(out_path, dpi=75, bbox_inches='tight')
    plt.close(fig)

    try:
        os.chmod(out_path, 0o770)
    except Exception as e:
        print(f"Warning: failed to set permissions for {out_path}: {e}")

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
    gif_frames[0].save(
        out_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )

    try:
        os.chmod(out_path, 0o770)
    except Exception as e:
        print(f"Warning: failed to set permissions for {out_path}: {e}")

    print("Saved animated GIF to:", out_path)
    return out_path
