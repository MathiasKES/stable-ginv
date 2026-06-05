"""Thin wrapper: moved to stable_ginv.viz.plot_normality_scatter (Phase 7).

Kept runnable as `python helper/plot_normality_scatter.py ...`.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_ginv.viz.plot_normality_scatter import main  # noqa: E402

if __name__ == "__main__":
    main()
