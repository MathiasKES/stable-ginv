"""Thin wrapper: moved to stable_ginv.viz.plot_rank_reconstruction (Phase 8).

Kept runnable as `python -m functions.rank_reconstruction_plot ...` and
`python functions/rank_reconstruction_plot.py ...`.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_ginv.viz.plot_rank_reconstruction import main  # noqa: E402

if __name__ == "__main__":
    main()
