"""Deterministic, CI-portable builders for golden characterization tests.

No dependency on /work3, downloaded datasets, or a GPU. Everything here is
seeded so the same inputs reproduce byte-for-byte across runs on one machine.
"""
import json
import os

import numpy as np

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
    """Faithful worker config dict (same keys as iDLG_mask.py builds).

    LeNet + no normalization keeps the recon path deterministic and free of
    dataset-normalization constants. METHODS='idlg' exercises the unmasked
    reconstruction only (masking is covered by the Tier 2 masking goldens).
    """
    return {
        "channel": 1,
        "num_classes": 10,
        "shape_img": (28, 28),
        "lr": 1.0,
        "num_dummy": 1,
        "Iteration": 10,
        "run_id": 0,
        "MASK_MODE": "gradsize_topfrac",   # unused when METHODS='idlg'
        "PREFIXES": (),
        "PREFIX_LAYER_FRACS": {},
        "GRADSIZE_TOPK": 20,
        "GRADSIZE_TOPFRAC": 0.5,
        "GRADSIZE_METRIC": "l2",
        "GRAD_LOSS": "cos",
        "GAMMA": 0.5,
        "NETWORK_NAME": "LeNet",
        "METHODS": method,
        "COMPUTE_JACOBIAN_RANK": False,
        "JACOBIAN_MAX_ENTRIES": 4000,
        "JACOBIAN_SELECT_MODE": "topk_abs",
        "TV_WEIGHT": 0.0,
        "OPTIMIZER": "lbfgs",
        "NUM_RESTARTS": 1,
        "MAX_ITERATION": 20,
        "HISTORY_SIZE": 100,
        "NETWORK_TRAINED": False,
        "SAVE_GIF": False,
        "FRAME_INTERVAL": 20,   # unused: SAVE_GIF=False, so no GIF frames are produced
        "out_path": None,
    }


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
