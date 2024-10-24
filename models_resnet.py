import re
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import *
from torchvision.models.resnet import Bottleneck


class ResNet(ResNet):
    def __init__(self, block, layers, **kwargs):
        super(ResNet, self).__init__(block, layers, **kwargs)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        features = x
        x = self.fc(x)

        return x, features


def _resnet(
    block,
    layers,
    weights,
    progress,
    **kwargs,
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    return model


def resnet50(*, weights=None, progress: bool = True, **kwargs):
    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)
