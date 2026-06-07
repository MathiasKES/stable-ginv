"""Animated reconstruction-progress GIF rendering.

Builds a per-iteration grid (rows = experiments, cols = [Init | iDLG? | Masked? |
GT]) and writes an animated GIF.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from stable_ginv.io import safe_chmod, safe_makedirs

sns.set_theme(style="whitegrid")


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
