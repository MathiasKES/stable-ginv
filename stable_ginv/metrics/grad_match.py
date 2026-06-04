import torch
import torch.nn.functional as F


def compute_grad_match_loss(dummy_dy_dx, selected_original, selected_entry_masks=None, grad_loss="cos", eps=1e-12):
    """Gradient matching loss between dummy and original gradients (l2 or cosine)."""
    grad_loss = grad_loss.lower()

    if grad_loss not in {"l2", "cos"}:
        raise ValueError(f"Unknown grad_loss: {grad_loss}")

    if grad_loss == "l2":
        total = 0.0
        num_terms = 0

        if selected_entry_masks is not None:
            for gx, gy, m in zip(dummy_dy_dx, selected_original, selected_entry_masks):
                gx_sel = gx[m]
                gy_sel = gy[m]
                if gx_sel.numel() == 0:
                    continue
                diff = gx_sel - gy_sel
                total = total + (diff ** 2).sum()
                num_terms += diff.numel()
        else:
            for gx, gy in zip(dummy_dy_dx, selected_original):
                diff = gx - gy
                total = total + (diff ** 2).sum()
                num_terms += diff.numel()

        return total, num_terms

    elif grad_loss == "cos":
        gx_all = []
        gy_all = []

        if selected_entry_masks is not None:
            for gx, gy, m in zip(dummy_dy_dx, selected_original, selected_entry_masks):
                gx_sel = gx[m]
                gy_sel = gy[m]
                if gx_sel.numel() == 0:
                    continue
                gx_all.append(gx_sel.reshape(-1))
                gy_all.append(gy_sel.reshape(-1))
        else:
            for gx, gy in zip(dummy_dy_dx, selected_original):
                gx_all.append(gx.reshape(-1))
                gy_all.append(gy.reshape(-1))

        if len(gx_all) == 0:
            raise ValueError("No gradient entries selected for cosine loss.")

        gx_cat = torch.cat(gx_all)
        gy_cat = torch.cat(gy_all)

        cos_sim = F.cosine_similarity(
            gx_cat.unsqueeze(0),
            gy_cat.unsqueeze(0),
            dim=1,
            eps=eps
        )

        return 1.0 - cos_sim[0], gx_cat.numel()
