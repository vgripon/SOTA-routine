from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

def Conv3x3(in_planes, out_planes, stride=1, groups=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=stride, groups=groups, bias=False)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1
    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = Conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = torch.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4
    def __init__(self, inplanes, width, stride, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = Conv3x3(width, width, stride)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, large_input, width, zero_init_residual=False):
        super().__init__()
        self.inplanes = width

        if large_input:
            self.embed = nn.Sequential(
                nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(self.inplanes),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            self.embed = nn.Sequential(
                nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.inplanes),
                nn.ReLU(inplace=True)
            )

        layers_list = []
        for depth, stride, multiplier in layers:
            layers_list.append(self._make_layer(block, width * multiplier, depth, stride=stride))
        self.layers = nn.Sequential(*layers_list)

        self.fc = nn.Linear(self.inplanes, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block, planes, blocks, stride = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride = 1,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.embed(x)

        x = self.layers(x)

        x = x.mean(-1).mean(-1)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def resnet18(num_classes, large_input, width):
    return ResNet(BasicBlock, [(2, 1, 1), (2, 2, 2), (2, 2, 4), (2, 2, 8)], num_classes, large_input, width)

def resnet34(num_classes, large_input, width):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, large_input, width)

def resnet50(num_classes, large_input, width):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, large_input, width)

def resnet101(num_classes, large_input, width):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, large_input, width)

def resnet152(num_classes, large_input, width):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, large_input, width)

def resnet20(num_classes, large_input, width):
    return ResNet(BasicBlock, [3, 3, 3], num_classes, large_input, width)

def resnet32(num_classes, large_input, width):
    return ResNet(BasicBlock, [5, 5, 5], num_classes, large_input, width)

def resnet44(num_classes, large_input, width):
    return ResNet(BasicBlock, [7, 7, 7], num_classes, large_input, width)

def resnet56(num_classes, large_input, width):
    return ResNet(BasicBlock, [9, 9, 9], num_classes, large_input, width)

def resnet110(num_classes, large_input, width):
    return ResNet(BasicBlock, [18, 18, 18], num_classes, large_input, width)

def resnet1202(num_classes, large_input, width):
    return ResNet(BasicBlock, [200, 200, 200], num_classes, large_input, width)
