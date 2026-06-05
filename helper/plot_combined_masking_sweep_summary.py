"""Thin wrapper: moved to stable_ginv.viz.plot_combined_masking_sweep_summary (Phase 7).

Kept runnable as `python helper/plot_combined_masking_sweep_summary.py ...`.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_ginv.viz.plot_combined_masking_sweep_summary import main  # noqa: E402

if __name__ == "__main__":
    main()
