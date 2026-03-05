import matplotlib.pyplot as plt
import os
import math
from Network import LeNet, LeNet_bigger, MediumCNN, weights_init

def get_keep_ids_by_gradsize(original_dy_dx, mode="topk", topk=10, top_frac=None, threshold=None, metric="l2"):
    """
    original_dy_dx: list of gradient tensors (same ordering as net.parameters()).
    mode:
      - "topk": keep the top-k largest gradient tensors
      - "topfrac": keep the top fraction (e.g. 0.3 means keep 30%)
      - "threshold": keep tensors with size >= threshold
    metric:
      - "l2": L2 norm per tensor
      - "mean_abs": mean absolute value per tensor
      - "sum_abs": sum of absolute values per tensor
    """
    sizes = []
    for i, g in enumerate(original_dy_dx):
        if g is None:
            sizes.append((i, float("-inf")))
            continue

        if metric == "l2":
            s = g.detach().norm(p=2).item()
        elif metric == "mean_abs":
            s = g.detach().abs().mean().item()
        elif metric == "sum_abs":
            s = g.detach().abs().sum().item()
        else:
            raise ValueError(f"Unknown metric: {metric}")

        sizes.append((i, s))

    # sort descending by size
    sizes_sorted = sorted(sizes, key=lambda x: x[1], reverse=True)

    if mode == "topk":
        k = min(topk, len(sizes_sorted))
        keep = [i for i, _ in sizes_sorted[:k]]

    elif mode == "topfrac":
        if top_frac is None:
            raise ValueError("top_frac must be set for mode='topfrac'")
        k = max(1, int(round(top_frac * len(sizes_sorted))))
        keep = [i for i, _ in sizes_sorted[:k]]

    elif mode == "threshold":
        if threshold is None:
            raise ValueError("threshold must be set for mode='threshold'")
        keep = [i for i, s in sizes_sorted if s >= threshold]
        if len(keep) == 0:
            # fallback: keep the single largest to avoid empty mask
            keep = [sizes_sorted[0][0]]

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return sorted(keep), sizes_sorted  # keep_ids, plus ranked list for optional logging

def build_network(name: str, channel: int, num_classes: int, input_size):
    if name == "LeNet":
        return LeNet(channel=channel, num_classes=num_classes, input_size=input_size)
    if name == "LeNet_bigger":
        return LeNet_bigger(channel=channel, num_classes=num_classes, input_size=input_size)
    if name == "MediumCNN":
        return MediumCNN(channel=channel, num_classes=num_classes, input_size=input_size)
    raise ValueError(f"Unknown NETWORK_NAME: {name}")

def get_keep_ids(mask_mode: str):
    if mask_mode == "all":
        return set(range(8))
    if mask_mode == "conv12":
        return {0, 1, 2, 3}
    if mask_mode == "conv123":
        return {0, 1, 2, 3, 4, 5}
    if mask_mode == "fc_only":
        return {6, 7}
    if mask_mode == "no_fc":
        return {0, 1, 2, 3, 4, 5}
    if mask_mode == "conv1_fc":
        return {0, 1, 6, 7}
    if mask_mode == "conv12_fc":
        return {0, 1, 2, 3, 6, 7}
    if mask_mode == "conv13_fc":
        return {0, 1, 4, 5, 6, 7}
    if mask_mode == "conv2_fc":
        return {2, 3, 6, 7}
    raise ValueError(f"Unknown MASK_MODE: {mask_mode}")

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
    os.chmod(out_path, 0o770) # Ensure correct permissions
    print("Saved 10-exp reconstruction panel to:", out_path)