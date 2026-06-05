"""Thin entry point: orchestration lives in stable_ginv.cli.batch (Phase 6).

Kept at the repo root so existing HPC commands (`python iDLG_mask.py ...`) and job
scripts keep working unchanged. New code should import from stable_ginv.cli.batch.
"""
from stable_ginv.cli.batch import main

if __name__ == '__main__':
    main()
