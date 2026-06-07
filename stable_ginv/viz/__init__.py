"""Visualization: reconstruction panels, animated GIFs, and restart curves.

Importing this package sets the Agg backend and the seaborn whitegrid theme as a
side effect.
"""
from stable_ginv.viz.panels import (
    append_result_to_panel_buffers,
    create_panel_buffers,
    flush_recon_panel,
    save_recon_panel,
)
from stable_ginv.viz.gif import save_recon_gif
from stable_ginv.viz.restart import (
    _gain_ci_str,
    _restart_stats,
    save_restart_curve,
    save_restart_images,
)

__all__ = [
    "append_result_to_panel_buffers",
    "create_panel_buffers",
    "flush_recon_panel",
    "save_recon_panel",
    "save_recon_gif",
    "save_restart_curve",
    "save_restart_images",
    "_restart_stats",
    "_gain_ci_str",
]
