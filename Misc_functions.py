import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import math
from Network import LeNet, LeNet_bigger, MediumCNN, BiggerCNN, get_model
import torch

def flatten_observed_gradients(grad_list, keep_ids=None, entry_masks=None):
    """
    Flatten observed gradients.
    Two modes:
    - tensor-wise masking via keep_ids
    - elementwise masking via entry_masks
    """
    flat = []
    for i, g in enumerate(grad_list):
        if g is None:
            continue
        if entry_masks is not None:
            m = entry_masks[i]
            if m is None:
                continue
            flat.append(g.reshape(-1)[m.reshape(-1)])
        else:
            if keep_ids is not None and i not in keep_ids:
                continue
            flat.append(g.reshape(-1))
    if len(flat) == 0:
        raise ValueError("No observed gradients selected.")
    return torch.cat(flat, dim=0)


def compute_jacobian_rank(net, x_norm, y, criterion, keep_ids=None, entry_masks=None):
    """
    Compute rank of J = d vec(g_obs(x)) / d vec(x),
    where g_obs(x) is the flattened observed gradient vector.
    """
    params = tuple(net.parameters())
    x_norm = x_norm.detach().clone().requires_grad_(True)

    def grad_vector(inp):
        out = net(inp)
        loss = criterion(out, y)
        grads = torch.autograd.grad(
            loss,
            params,
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )
        return flatten_observed_gradients(grads, keep_ids=keep_ids, entry_masks=entry_masks)

    J = torch.autograd.functional.jacobian(grad_vector, x_norm, vectorize=True)
    J = J.reshape(J.shape[0], -1)

    observed_entries = J.shape[0]
    unknowns = J.shape[1]
    jac_rank = int(torch.linalg.matrix_rank(J).item())

    return jac_rank, tuple(J.shape), observed_entries, unknowns


def get_entry_masks_by_gradsize(original_dy_dx, mode="topk_entries", topk=None, top_frac=None, candidate_ids=None):
    """
    Elementwise masking:
    - topk_entries: keep exactly topk scalar gradient entries by absolute magnitude
    - topfrac_entries: keep exactly round(top_frac * total_entries) scalar entries

    Returns:
        entry_masks: list of boolean tensors matching original_dy_dx
        kept_entries: exact number of kept scalar entries
        total_entries: total number of scalar entries
    """
    pieces = []
    meta = []

    for i, g in enumerate(original_dy_dx):
        if candidate_ids is not None and i not in candidate_ids:
            meta.append((i, 0, None))
            continue
        if g is None:
            meta.append((i, 0, None))
            continue
        flat = g.detach().abs().reshape(-1)
        pieces.append(flat)
        meta.append((i, flat.numel(), g.shape))

    if len(pieces) == 0:
        raise ValueError("No gradients available for masking.")

    all_vals = torch.cat(pieces, dim=0)
    total_entries = all_vals.numel()

    if mode == "topk_entries":
        if topk is None:
            raise ValueError("topk must be set for mode='topk_entries'")
        k = max(1, min(int(topk), total_entries))
    elif mode == "topfrac_entries":
        if top_frac is None:
            raise ValueError("top_frac must be set for mode='topfrac_entries'")
        k = max(1, min(int(round(top_frac * total_entries)), total_entries))
    else:
        raise ValueError(f"Unknown entrywise mode: {mode}")

    top_idx = torch.topk(all_vals, k=k, largest=True).indices
    global_mask = torch.zeros(total_entries, dtype=torch.bool, device=all_vals.device)
    global_mask[top_idx] = True

    entry_masks = []
    offset = 0
    for i, numel, shape in meta:
        g = original_dy_dx[i]

        if g is None or shape is None or numel == 0:
            entry_masks.append(None)
            continue

        local_mask = global_mask[offset:offset + numel].reshape(shape)
        entry_masks.append(local_mask)
        offset += numel

    return entry_masks, k, total_entries

def get_keep_ids_by_gradsize(original_dy_dx, mode="topk", topk=10, top_frac=None, threshold=None, metric="l2", candidate_ids=None):
    """
    Tensor-wise masking.
    """
    sizes = []
    for i, g in enumerate(original_dy_dx):
        if candidate_ids is not None and i not in candidate_ids:
            continue
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
            keep = [sizes_sorted[0][0]]

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return sorted(keep), sizes_sorted

def build_network(name: str, channel: int, num_classes: int, input_size):
    if name == "LeNet":
        return LeNet(channel=channel, num_classes=num_classes, input_size=input_size)
    if name == "LeNet_bigger":
        return LeNet_bigger(channel=channel, num_classes=num_classes, input_size=input_size)
    if name == "MediumCNN":
        return MediumCNN(channel=channel, num_classes=num_classes, input_size=input_size)
    if name == "BiggerCNN":
        return BiggerCNN(channel=channel, num_classes=num_classes, input_size=input_size)
    if name.lower().startswith("resnet"): # 18, 34, 50, 101, 152
        return get_model(network=name.lower(), channel=channel, num_classes=num_classes, input_size=input_size)
    raise ValueError(f"Unknown NETWORK_NAME: {name}")

def get_prefix_keep_ids(net, prefixes):
    keep = set()
    for idx, (name, _) in enumerate(net.named_parameters()):
        if any(name.startswith(p) for p in prefixes):
            keep.add(idx)
    if len(keep) == 0:
        raise ValueError(f"No parameters matched prefixes={prefixes}")
    return keep

def get_keep_ids(mask_mode: str, net=None, prefixes=None):
    """
    Returns a set of parameter indices to keep.

    - If prefixes is provided: uses net.named_parameters() and keeps any parameter whose
      name starts with one of the prefixes.
    - Otherwise uses your legacy index-based masks (works for LeNet/MediumCNN etc.).

    Example:
      keep_ids = get_keep_ids(mask_mode="prefix", net=net, prefixes=("conv1","layer1","linear"))
    """

    # --- prefix-based mode (model-agnostic) ---
    if prefixes is not None:
        if net is None:
            raise ValueError("prefix-based keep_ids requires 'net'")
        keep = set()
        for idx, (name, _) in enumerate(net.named_parameters()):
            if any(name.startswith(p) for p in prefixes):
                keep.add(idx)
        if len(keep) == 0:
            raise ValueError(f"No parameters matched prefixes={prefixes}")
        return keep
    
    if mask_mode == "all":
        if net is None:
            return set(range(8))  # legacy behavior
        return set(range(len(list(net.parameters()))))  # works for any net if provided

    raise ValueError(f"Unknown MASK_MODE: {mask_mode}")

def compute_psnr_from_mse(mse: float, max_val: float = 1.0, eps: float = 1e-12) -> float:
    """Compute PSNR in dB from a scalar MSE. Assumes images are in [0, max_val]."""
    mse = float(mse)
    if mse < eps:
        return float('inf')
    return 10.0 * math.log10((max_val * max_val) / (mse + eps))

def save_recon_panel(params: dict, panel_gt_pil, panel_idlg_pil, panel_masked_pil,
                     save_dir, block_idx, dataset, mask_desc: str, timestamp_str: str,
                     methods: str = "both"):
    """
    Save a panel with dynamic rows depending on methods:

    methods="both"   -> rows: GT, iDLG, iDLG_masked
    methods="idlg"   -> rows: GT, iDLG
    methods="masked" -> rows: GT, iDLG_masked
    """
    n = len(panel_gt_pil)
    if n == 0:
        return

    rows = [("GT", panel_gt_pil)]

    if methods in ["idlg", "both"]:
        rows.append(("iDLG", panel_idlg_pil))

    if methods in ["masked", "both"]:
        rows.append(("iDLG_masked", panel_masked_pil))

    num_rows = len(rows)
    fig = plt.figure(figsize=(2.2 * n + 1.6, 2.1 * num_rows))

    # Row labels + images
    for r, (row_name, row_imgs) in enumerate(rows):
        # y-position for label from top to bottom
        y = 1.0 - (r + 0.5) / num_rows
        fig.text(0.01, y, row_name, va='center', ha='left',
                 fontsize=14, fontweight='bold')

        for j in range(n):
            ax = plt.subplot(num_rows, n, r * n + 1 + j)
            ax.imshow(row_imgs[j], cmap='gray' if dataset == 'MNIST' else None)
            if r == 0:
                ax.set_title(f"exp {j}")
            ax.axis('off')

    plt.tight_layout(rect=(0.08, 0.0, 1.0, 1.0))
    out_path = os.path.join(
        save_dir,
        f"{'_'.join(f'{key}{val}' for key, val in params.items())}_{mask_desc}_{methods}_{timestamp_str}.png"
    )
    plt.savefig(out_path, dpi=250, bbox_inches='tight')
    plt.close(fig)

    try:
        os.chmod(out_path, 0o770)
    except Exception as e:
        print(f"Warning: failed to set permissions for {out_path}: {e}")

    print("Saved reconstruction panel to:", out_path)