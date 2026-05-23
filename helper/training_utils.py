import torch


def make_scheduler(optimizer, iteration, gamma):
    """MultiStepLR with milestones at 3/8, 5/8, 7/8 of total iterations."""
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[
            int(iteration * 3 / 8),
            int(iteration * 5 / 8),
            int(iteration * 7 / 8),
        ],
        gamma=gamma,
    )
