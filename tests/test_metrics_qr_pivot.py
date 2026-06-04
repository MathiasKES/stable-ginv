import subprocess
import sys

import pytest
import torch

from stable_ginv.metrics.jacobian import _load_scipy_linalg, _qr_pivot_rows


def test_metrics_import_does_not_eagerly_load_scipy_linalg():
    # scipy.linalg must stay out of worker startup: a normal run does no QR
    # pivoting, and importing scipy.linalg can fail on HPC with an old system
    # C++ runtime. (skimage pulls in scipy core but not scipy.linalg.)
    for target in ("stable_ginv.metrics", "helper.metrics"):
        code = f"import {target}, sys; print('scipy.linalg' in sys.modules)"
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert out.stdout.strip() == "False", f"{target} eagerly loaded scipy.linalg"


def test_load_scipy_linalg_is_cached():
    pytest.importorskip("scipy.linalg")
    assert _load_scipy_linalg() is _load_scipy_linalg()


def test_qr_pivot_rows_orders_by_leading_pivot():
    pytest.importorskip("scipy.linalg")
    # QR column pivoting on J^T selects the largest-norm row of J first.
    J = torch.tensor([[1.0, 0.0], [0.0, 3.0], [2.0, 0.0]])
    out = _qr_pivot_rows(J)
    assert out.shape == J.shape
    assert torch.equal(out[0], J[1])           # row 1 has the largest norm (3)
    assert sorted(out.tolist()) == sorted(J.tolist())  # a row permutation of J
