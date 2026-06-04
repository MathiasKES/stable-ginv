"""One-time label inference from the original unmasked final-FC weight gradient."""
import torch

from stable_ginv.masking import _get_last_fc_param_indices


class LabelInference:
    """Predict the ground-truth label from the original gradient (iDLG label trick).

    Uses the sign of the row-summed final fully-connected weight gradient. This is
    computed once from the *unmasked* gradient and shared across both the iDLG and
    masked reconstructions.
    """

    @staticmethod
    def infer(net, original_dy_dx):
        """Return the predicted label as a 1-element long tensor."""
        named_params_list = list(net.named_parameters())
        last_fc_ids = _get_last_fc_param_indices(net)
        final_weight_idx = next(
            i for i in sorted(last_fc_ids) if named_params_list[i][0].endswith(".weight")
        )
        return torch.argmin(
            torch.sum(original_dy_dx[final_weight_idx], dim=-1), dim=-1
        ).detach().reshape((1,))
