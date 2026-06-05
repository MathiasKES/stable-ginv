"""Thin wrapper: moved to stable_ginv.viz.plot_paired_masking_violin (Phase 7).

Kept runnable as `python helper/plot_paired_masking_violin.py ...`.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_ginv.viz.plot_paired_masking_violin import main  # noqa: E402

if __name__ == "__main__":
    main()
