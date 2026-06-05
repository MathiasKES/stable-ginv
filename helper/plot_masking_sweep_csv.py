"""Thin wrapper: moved to stable_ginv.viz.plot_masking_sweep_csv (Phase 7).

Kept runnable as `python helper/plot_masking_sweep_csv.py <sweep_csv>` (see README).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_ginv.viz.plot_masking_sweep_csv import main  # noqa: E402

if __name__ == "__main__":
    main()
