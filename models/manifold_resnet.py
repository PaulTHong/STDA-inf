"""
PyTorch implementation for ResNet with Manifold Mixup.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import sys
sys.path.insert(1, '/home/hongt/STDA-inf')
from utils import mixup_data


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # kernel_size=3, stride=1
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, target=None, mixup_hidden=False, mixup_alpha=0.1, layer_mix=None):
        if mixup_hidden:
            if layer_mix is None:
                layer_mix = random.randint(0, 2)  # 0,1,2
            else:
                layer_mix = random.choice(layer_mix)
            out = x
            if layer_mix == 0:
                out, y_a, y_b, lam = mixup_data(out, target, mixup_alpha)
            
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            if layer_mix == 1:
                # out = lam * out + (1 - lam) * out[index,:]
                out, y_a, y_b, lam = mixup_data(out, target, mixup_alpha)

            out = self.layer2(out)
            if layer_mix == 2:
                out, y_a, y_b, lam = mixup_data(out, target, mixup_alpha)

            out = self.layer3(out)
            if layer_mix == 3:
                out, y_a, y_b, lam = mixup_data(out, target, mixup_alpha)

            out = self.layer4(out)
            if layer_mix == 4:
                out, y_a, y_b, lam = mixup_data(out, target, mixup_alpha)

            out = F.avg_pool2d(out, 6)
            out = out.view(out.size(0), -1)
            out = self.linear(out)

            if layer_mix == 5:
                out, y_a, y_b, lam = mixup_data(out, target, mixup_alpha)
            lam = torch.tensor(lam).to(x.device)
            lam = lam.repeat(y_a.size())
            return out, y_a, y_b, lam
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, 6)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50(num_classes=10):
    return ResNet(Bottleneck, [3,4,6,3], num_classes)

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])

def ManifoldResNet50(num_classes=10):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes)
    return model


def test():
    net = ResNet50()
    y = net(torch.randn(1, 3, 96, 96))
    print(y.size())

# test()
