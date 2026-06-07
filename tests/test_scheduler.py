"""Locks make_scheduler's MultiStepLR milestones/gamma.

make_scheduler is not covered by the recon-worker golden (that golden uses
optimizer='lbfgs', which never builds a scheduler), so this test guards the
scheduler factory in stable_ginv/recon/scheduler.py.
"""
import torch

from stable_ginv.recon.scheduler import make_scheduler


def _scheduler_for(iteration, gamma=0.5):
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=1.0)
    return make_scheduler(optimizer, iteration, gamma=gamma)


def test_make_scheduler_milestones_and_gamma():
    sched = _scheduler_for(iteration=1000, gamma=0.5)
    assert isinstance(sched, torch.optim.lr_scheduler.MultiStepLR)
    assert sched.milestones == {375: 1, 625: 1, 875: 1}
    assert sched.gamma == 0.5


def test_make_scheduler_milestones_floor_division():
    # 3/8, 5/8, 7/8 of 100 -> int() floors to 37, 62, 87.
    sched = _scheduler_for(iteration=100, gamma=0.1)
    assert sched.milestones == {37: 1, 62: 1, 87: 1}
    assert sched.gamma == 0.1
