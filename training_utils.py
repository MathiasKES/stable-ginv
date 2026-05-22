import torch
from Network import LeNet, LeNet_bigger, MediumCNN, BiggerCNN, get_model


def build_network(name: str, channel: int, num_classes: int, input_size, pretrained=False):
    """Instantiate a model by name; supports custom CNNs and torchvision backbones."""
    if name == "LeNet":
        return LeNet(channel=channel, num_classes=num_classes, input_size=input_size)
    if name == "LeNet_bigger":
        return LeNet_bigger(channel=channel, num_classes=num_classes, input_size=input_size)
    if name == "MediumCNN":
        return MediumCNN(channel=channel, num_classes=num_classes, input_size=input_size)
    if name == "BiggerCNN":
        return BiggerCNN(channel=channel, num_classes=num_classes, input_size=input_size)
    if name.lower().startswith(("resnet", "resnext", "wide_resnet", "vgg", "densenet")):
        return get_model(
            network=name.lower(),
            channel=channel,
            num_classes=num_classes,
            input_size=input_size,
            pretrained=pretrained
        )
    raise ValueError(f"Unknown NETWORK_NAME: {name}")


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
