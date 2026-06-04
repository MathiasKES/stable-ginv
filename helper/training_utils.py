"""Shim: the LR scheduler factory lives in stable_ginv.recon.scheduler (Phase 5)."""
from stable_ginv.recon.scheduler import make_scheduler

__all__ = ["make_scheduler"]
