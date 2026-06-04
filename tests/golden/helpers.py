"""Deterministic, CI-portable builders for golden characterization tests.

No dependency on /work3, downloaded datasets, or a GPU. Everything here is
seeded so the same inputs reproduce byte-for-byte across runs on one machine.
"""
import json
import os

import numpy as np
from stable_ginv.config import ExperimentConfig

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

# Set GOLDEN_REGEN=1 to (re)write fixtures instead of asserting against them.
REGEN = os.environ.get("GOLDEN_REGEN") == "1"


class FakeImageDataset:
    """Minimal dataset matching the worker's `dst[i] -> (HxWxC uint8 array, int)`.

    torchvision ToTensor() converts an (H, W, C) uint8 array to a [C, H, W]
    float tensor in [0, 1], which is exactly what run_single_exp expects.
    """

    def __init__(self, n, channel, size, num_classes, seed=0):
        rng = np.random.default_rng(seed)
        h, w = size
        self.images = [
            rng.integers(0, 256, size=(h, w, channel), dtype=np.uint8) for _ in range(n)
        ]
        self.labels = [int(rng.integers(0, num_classes)) for _ in range(n)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.images[i], self.labels[i]


def lenet_worker_config(method="idlg"):
    """Faithful worker config (ExperimentConfig) for golden characterization tests.

    LeNet + no normalization keeps the recon path deterministic and free of
    dataset-normalization constants. method='idlg' exercises the unmasked
    reconstruction only.
    """
    return ExperimentConfig(
        channel=1,
        num_classes=10,
        shape_img=(28, 28),
        lr=1.0,
        num_dummy=1,
        iteration=10,
        run_id=0,
        mask_mode="gradsize_topfrac",
        prefixes=(),
        prefix_layer_fracs={},
        gradsize_topk=20,
        gradsize_topfrac=0.5,
        gradsize_metric="l2",
        grad_loss="cos",
        gamma=0.5,
        network_name="LeNet",
        methods=method,
        compute_jacobian_rank=False,
        jacobian_max_entries=4000,
        jacobian_select_mode="topk_abs",
        tv_weight=0.0,
        optimizer="lbfgs",
        num_restarts=1,
        max_iteration=20,
        history_size=100,
        network_trained=False,
        save_gif=False,
        frame_interval=20,
        out_path=None,
    )


def load_or_regen(name, produce):
    """Return the golden fixture `name`, regenerating it when GOLDEN_REGEN=1.

    `produce` is a zero-arg callable returning a JSON-serializable object.
    """
    path = os.path.join(FIXTURES_DIR, name)
    if REGEN or not os.path.exists(path):
        value = produce()
        with open(path, "w") as f:
            json.dump(value, f, indent=2, sort_keys=True)
        return value
    with open(path) as f:
        return json.load(f)
