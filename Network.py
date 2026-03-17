import torch.nn as nn
import torch
from torchvision import models

def get_model(network, channel=3, num_classes=10, input_size=(32,32)):
    # model = models.resnet18(weights=None)
    model = getattr(models, network)(weights=None)
    if channel != 3:
        model.conv1 = nn.Conv2d(
            channel, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

class LeNet(nn.Module):
    def __init__(self, channel=3, num_classes=10, input_size=(32,32)):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, channel, input_size[0], input_size[1])
            feat = self.body(dummy)
            hidden = feat.view(1, -1).size(1)
            
        self.fc = nn.Sequential(
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class LeNet_bigger(nn.Module):
    def __init__(self, channel=3, num_classes=10, input_size=(32,32)):
        super().__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(channel, 16, kernel_size=5, padding=2, stride=2),  # wider + no early downsample
            act(),
            nn.Conv2d(16, 32, kernel_size=5, padding=2, stride=2),      # downsample here
            act(),
            nn.Conv2d(32, 32, kernel_size=5, padding=1, stride=1),
            act(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, channel, input_size[0], input_size[1])
            feat = self.body(dummy)
            hidden = feat.view(1, -1).size(1)

        self.fc = nn.Linear(hidden, num_classes)  # because stride=2 once: 32->16

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        return self.fc(out)

class MediumCNN(nn.Module):
    def __init__(self, channel=3, num_classes=10, input_size=(32, 32)):
        super().__init__()
        act = nn.Sigmoid

        self.body = nn.Sequential(
            nn.Conv2d(channel, 32, kernel_size=3, padding=1, stride=1),
            act(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),   # 32 -> 16
            act(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),  # 16 -> 8
            act(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1), # 8 -> 8
            act(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, channel, input_size[0], input_size[1])
            feat = self.body(dummy)
            hidden = feat.view(1, -1).size(1)

        # moderate head (you can mask fc separately)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = self.body(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class BiggerCNN(nn.Module):
    def __init__(self, channel=3, num_classes=10, input_size=(32, 32)):
        super().__init__()
        act = nn.Sigmoid

        self.body = nn.Sequential(
            nn.Conv2d(channel, 64, kernel_size=3, padding=1, stride=1),   # 32 -> 32
            act(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),        # 32 -> 32
            act(),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),       # 32 -> 16
            act(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),      # 16 -> 16
            act(),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),      # 16 -> 8
            act(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),      # 8 -> 8
            act(),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),      # 8 -> 8
            act(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, channel, input_size[0], input_size[1])
            feat = self.body(dummy)
            hidden = feat.view(1, -1).size(1)

        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = self.body(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def weights_init(m):
    try:
        if hasattr(m, "weight") and m.weight is not None:
            m.weight.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.weight' % m._get_name())
    try:
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.bias' % m._get_name())