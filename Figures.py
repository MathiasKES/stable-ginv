import matplotlib.pyplot as plt
import os
import math

def compute_psnr_from_mse(mse: float, max_val: float = 1.0, eps: float = 1e-12) -> float:
    """Compute PSNR in dB from a scalar MSE. Assumes images are in [0, max_val]."""
    mse = float(mse)
    if mse < eps:
        return float('inf')
    return 10.0 * math.log10((max_val * max_val) / (mse + eps))


def save_recon_panel(params: dict, panel_gt_pil, panel_idlg_pil, panel_masked_pil, save_dir, block_idx, dataset, mask_desc: str, timestamp_str: str):
    """
    Save a 3xN panel:
      Row 1: Ground truth images
      Row 2: iDLG final recon
      Row 3: iDLG_masked final recon
    """
    n = len(panel_gt_pil)
    if n == 0:
        return

    fig = plt.figure(figsize=(2.2 * n + 1.6, 6.2))

    # Row labels (left side)
    fig.text(0.01, 0.83, 'GT', va='center', ha='left', fontsize=14, fontweight='bold')
    fig.text(0.01, 0.50, 'iDLG', va='center', ha='left', fontsize=14, fontweight='bold')
    fig.text(0.01, 0.17, 'iDLG_masked', va='center', ha='left', fontsize=14, fontweight='bold')
    
    # Row 1: GT
    for j in range(n):
        ax = plt.subplot(3, n, 1 + j)
        ax.imshow(panel_gt_pil[j], cmap='gray' if dataset == 'MNIST' else None)
        ax.set_title(f"exp {j}")
        ax.axis('off')

    # Row 2: iDLG
    for j in range(n):
        ax = plt.subplot(3, n, 1 + n + j)
        ax.imshow(panel_idlg_pil[j], cmap='gray' if dataset == 'MNIST' else None)
        ax.axis('off')

    # Row 3: iDLG_masked
    for j in range(n):
        ax = plt.subplot(3, n, 1 + 2 * n + j)
        ax.imshow(panel_masked_pil[j], cmap='gray' if dataset == 'MNIST' else None)
        ax.axis('off')

        # Leave space on the left for row labels
        plt.tight_layout(rect=(0.08, 0.0, 1.0, 1.0))
        out_path = os.path.join(save_dir, f"{'_'.join(f'{key}{val}' for key, val in params.items())}_{mask_desc}_{timestamp_str}.png")
        plt.savefig(out_path, dpi=250, bbox_inches='tight')
        plt.close(fig)
        print("Saved 10-exp reconstruction panel to:", out_path)