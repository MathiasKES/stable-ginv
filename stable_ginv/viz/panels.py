"""Reconstruction-panel buffers and adaptive-row PNG panel rendering.

Holds per-experiment panel buffers, orders them by experiment index on flush, and
writes the PNG panel whose rows adapt to which methods (iDLG / masked) were run.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from stable_ginv.io import safe_savefig

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
