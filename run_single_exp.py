"""Worker shim: reconstruction lives in stable_ginv.recon (Phase 5).

Kept importable at this path so the multiprocessing worker target spawned by
iDLG_mask.py (`from run_single_exp import run_single_experiment`) and the
recon-worker golden continue to work unchanged. New code should import from
stable_ginv.recon.
"""
from stable_ginv.recon import run_single_experiment, _run_inner

__all__ = ["run_single_experiment", "_run_inner"]
