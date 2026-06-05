"""Thin wrapper: moved to stable_ginv.jacobian (Phase 8).

Kept runnable as `python functions/jacobian_rank_sweep.py ...` and
`python -m functions.jacobian_rank_sweep ...`. The mp.spawn worker now lives at
the stable import path stable_ginv.jacobian.sweep._mp_worker.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch.multiprocessing as mp  # noqa: E402
from stable_ginv.jacobian.cli import main  # noqa: E402

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
