"""ExperimentConfig: frozen dataclass replacing the UPPERCASE config dict."""
from __future__ import annotations

import dataclasses
from typing import Optional


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    """Frozen, picklable experiment configuration.

    Carries every per-experiment setting from the CLI into the reconstruction
    worker. Dataset-derived fields (channel, num_classes, shape_img) have no
    defaults and must be provided from load_dataset().  All other defaults match
    idlg_cli.py argument defaults.

    single_restart_idx is None in the base config; parallel-restart dispatch
    sets it via dataclasses.replace(config, single_restart_idx=r_i).
    """

    # --- dataset-derived (required — set by load_dataset) ---
    channel: int
    num_classes: int
    shape_img: tuple          # (H, W)

    # --- CLI-derived (defaults match idlg_cli.py) ---
    lr: float = 1.0
    num_dummy: int = 1
    iteration: int = 1000
    run_id: int = 0
    mask_mode: str = "gradsize_topfrac"
    prefixes: tuple = ()
    prefix_layer_fracs: dict = dataclasses.field(default_factory=dict)
    gradsize_topk: Optional[int] = 20
    gradsize_topfrac: float = 0.5
    gradsize_metric: str = "l2"
    grad_loss: str = "cos"
    gamma: float = 0.5
    network_name: str = "LeNet"
    methods: str = "idlg"
    compute_jacobian_rank: bool = False
    jacobian_max_entries: int = 4000
    jacobian_select_mode: str = "topk_abs"
    tv_weight: float = 0.0
    optimizer: str = "lbfgs"
    num_restarts: int = 1
    max_iteration: int = 20
    history_size: int = 100
    network_trained: bool = False
    save_gif: bool = False
    frame_interval: int = 20
    out_path: Optional[str] = None
    single_restart_idx: Optional[int] = None
