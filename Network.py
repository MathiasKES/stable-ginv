import torch.nn as nn
import torch
import torch.nn.init as init
from torchvision.models import resnet18
import torch.nn.functional as F
import torchvision
from torchvision.models.resnet import BasicBlock, Bottleneck

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

import torch
import torch.nn as nn

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

###########################

class ResNetCIFAR(nn.Module):
    """
    CIFAR-style ResNet equivalent to the invertinggradients repo's ResNet wrapper:
      - conv1: 3x3, stride 1, pad 1, bias False
      - stages: widths base_width, 2*base_width, 4*base_width, ... (for CIFAR: 3 stages)
      - downsample: torchvision BasicBlock downsample (1x1 conv + BN) when stride != 1 or channels mismatch
      - pool: AdaptiveAvgPool2d((1,1)) by default
      - fc: Linear(width_last * expansion, num_classes)
      - init: kaiming_normal fan_out relu + BN weight=1 bias=0
    """

    def __init__(self,
                 block=BasicBlock,
                 layers=(3, 3, 3),          # ResNet20 on CIFAR
                 num_classes=10,
                 num_channels=3,
                 base_width=16,
                 strides=(1, 2, 2),         # CIFAR: stage strides
                 pool='avg',
                 zero_init_residual=False,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()
        self._norm_layer = norm_layer

        self.inplanes = base_width
        self.conv1 = nn.Conv2d(num_channels, self.inplanes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        # Build CIFAR stages (3 stages for ResNet20/32/44/56/110)
        self.layers = nn.ModuleList()
        width = self.inplanes
        for idx, num_blocks in enumerate(layers):
            stride = strides[idx]
            self.layers.append(self._make_layer(block, width, num_blocks, stride=stride))
            width *= 2

        self.pool = nn.AdaptiveAvgPool2d((1, 1)) if pool == 'avg' else nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear((width // 2) * block.expansion, num_classes)

        self._init_weights(zero_init_residual=zero_init_residual)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None

        # torchvision BasicBlock expects downsample when shape changes
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample,
                            groups=1, base_width=64, dilation=1, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, downsample=None,
                                groups=1, base_width=64, dilation=1, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _init_weights(self, zero_init_residual=False):
        # Match invertinggradients init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        for layer in self.layers:
            x = layer(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def resnet20(channel=3, num_classes=10):
    return ResNetCIFAR(block=BasicBlock, layers=(3, 3, 3),
                          num_classes=num_classes, num_channels=channel, base_width=16)

#########################

def weights_init_adv(m):
    """
        Initialization of CNN weights
    """
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

# We define all the classes and function regarding the ResNet architecture in this code cell
__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

class LambdaLayer(nn.Module):
    """
      Identity mapping between ResNet blocks with diffrenet size feature map
    """
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

# A basic block as shown in Fig.3 (right) in the paper consists of two convolutional blocks, each followed by a Bach-Norm layer. 
# Every basic block is shortcuted in ResNet architecture to construct f(x)+x module. 
# Expansion for option 'A' in the paper is equal to identity with extra zero entries padded
# for increasing dimensions between layers with different feature map size. This option introduces no extra parameter. 
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 experiment, ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# Stack of 3 times 2*n (n is the number of basic blocks) layers are used for making the ResNet model, 
# where each 2n layers have feature maps of size {16,32,64}, respectively. 
# The subsampling is performed by convolutions with a stride of 2.
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, channel=3, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(channel, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)
        self.apply(weights_init_adv)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(channel=3, num_classes=10):
    return ResNet(BasicBlock, [3, 3, 3], channel=channel, num_classes=num_classes)