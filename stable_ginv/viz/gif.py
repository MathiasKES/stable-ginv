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

    cols = []
    if show_idlg:
        cols.append("iDLG")
    if show_masked:
        cols.append("masked")
    cols.append("GT")
    n_cols = len(cols)

    both = show_idlg and show_masked
    col_title = {
        #"init": "Random Init",
        "iDLG": "Unmasked" if both else "Recovered",
        "masked": "Masked" if both else "Recovered",
        "GT": "Ground Truth",
    }

    # total_frames = max_frames + 5

    # gif_frames = []
    # for t in range(total_frames):
    #     frame_t = min(t, max_frames - 1)

    max_gif_frames = None   # cap rendered frames (set None to show all)
    tail_frames = 40       # hold the final frame this many extra frames
    if max_gif_frames is not None and max_frames > max_gif_frames:
        sel = np.linspace(0, max_frames - 1, max_gif_frames)
        frame_indices = sorted({int(round(i)) for i in sel})
    else:
        frame_indices = list(range(max_frames))
    if frame_indices:
        frame_indices += [frame_indices[-1]] * max(0, tail_frames)

    gif_frames = []
    for frame_t in frame_indices:

        fig_w = max(4.0, 1.8 * n_cols + 0.6)
        fig_h = max(3.0, 1.8 * n_exp + 0.4)
        fig, axes = plt.subplots(
            n_exp, n_cols, figsize=(fig_w, fig_h),
            squeeze=False,
            gridspec_kw={'hspace': 0.2, 'wspace': 0},
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
            # y_pos = 1.0 - (row_idx + 0.5) / n_exp
            # fig.text(0.005, y_pos, f"Exp {row_idx}", va='center', ha='left',
            #          fontsize=6, fontweight='bold')

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

        #plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        fig.subplots_adjust(left=0.02, right=0.88, top=0.91, bottom=0.08, hspace=0.2, wspace=0)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=125)
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
    # try:
    #     gif_frames[0].save(
    #         out_path,
    #         save_all=True,
    #         append_images=gif_frames[1:],
    #         duration=duration_ms,
    #         loop=0,
    #         optimize=False,
    #     )
    # except OSError as e:
    # GIF allows only 256 colors/frame. Quantizing each frame independently (with
    # dithering) makes identical pixels — like the static GT — map to different
    # palette colors each frame, which looks like flickering. Build ONE shared
    # palette and map every frame to it with no dithering so static regions are stable.
    rgb_frames = [f.convert("RGB") for f in gif_frames]
    palette = rgb_frames[-1].quantize(colors=256, method=PILImage.MEDIANCUT,
                                      dither=PILImage.NONE)  # final frame has the true colors
    pal_frames = [f.quantize(palette=palette, dither=PILImage.NONE) for f in rgb_frames]
    try:
        pal_frames[0].save(
            out_path,
            save_all=True,
            append_images=pal_frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
            disposal=1,
        )
    except OSError as e:
        print(f"[WARNING] Failed to save GIF {out_path}: {e}")
        return None

    safe_chmod(out_path)
    print("Saved animated GIF to:", out_path)
    return out_path
