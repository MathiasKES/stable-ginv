# Re-export shim — all functions have moved to focused modules.
# This file exists only for any external scripts that still import from Misc_functions.
# Remove once all import sites are confirmed updated.

from masking import (
    flatten_observed_gradients,
    build_gradient_mask,
    get_keep_ids,
    get_keep_ids_by_gradsize,
    get_entry_masks_by_gradsize,
    get_prefix_keep_ids,
    get_entry_masks_by_prefix_group,
    get_keep_ids_by_prefix_group,
)
from metrics import (
    compute_psnr_from_mse,
    compute_ssim_batch,
    total_variation,
    compute_jacobian_rank,
    compute_grad_match_loss,
)
from visualization import save_recon_panel, save_recon_gif
from io_utils import (
    paired_t_ci,
    paired_summary,
    baseline_key_from_args,
    load_baseline_registry,
    save_baseline_registry,
    update_idlg_baseline,
    write_baseline_summary_csv,
)
from training_utils import build_network, make_scheduler
