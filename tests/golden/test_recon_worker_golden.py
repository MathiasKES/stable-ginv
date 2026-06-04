"""Golden test: in-process reconstruction worker numerics.

Calls stable_ginv.recon._run_inner directly on CPU with a fixed seed/config and a
synthetic dataset, capturing the result dict via a stub queue. Locks recon
metrics so later phases prove no numeric drift.

Tolerance: integer/label/shape fields are exact; floating metrics use rel=1e-4
to absorb platform float noise while still catching real logic changes (which
move metrics far more than 1e-4). If the torch/env version changes, regenerate:
    GOLDEN_REGEN=1 pytest tests/golden/test_recon_worker_golden.py
"""
import pytest

from stable_ginv.recon import _run_inner
from tests.golden.helpers import FakeImageDataset, lenet_worker_config, load_or_regen


class _CaptureQueue:
    """Stand-in for mp.SimpleQueue. _run_inner is expected to put() exactly once."""

    def __init__(self):
        self.result = None

    def put(self, value):
        assert self.result is None, "_run_inner called put() more than once — golden captured an unexpected payload"
        self.result = value


def _run_worker():
    dst = FakeImageDataset(16, channel=1, size=(28, 28), num_classes=10, seed=0)
    config = lenet_worker_config(method="idlg")
    queue = _CaptureQueue()
    _run_inner(0, 0, dst, "MNIST", config, queue)
    return queue.result


def _produce():
    r = _run_worker()
    return {
        "label_iDLG": r["label_iDLG"],
        "loss_iDLG": r["best_loss_iDLG"],
        "mse_iDLG": r["best_mse_iDLG"],
        "psnr_idlg": r["best_psnr_idlg"],
        "early_stop_reason": r["early_stop_reason"].get("iDLG"),
    }


def test_recon_worker_numerics_stable():
    golden = load_or_regen("recon_worker_golden.json", _produce)
    current = _produce()
    assert current["label_iDLG"] == golden["label_iDLG"]
    assert current["early_stop_reason"] == golden["early_stop_reason"]
    assert current["loss_iDLG"] == pytest.approx(golden["loss_iDLG"], rel=1e-4, abs=1e-8), "loss_iDLG"
    for k in ("mse_iDLG", "psnr_idlg"):
        assert current[k] == pytest.approx(golden[k], rel=1e-4), k
