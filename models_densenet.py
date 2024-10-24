"""Pytorch Densenet implementation w/ tweaks
This file is a copy of https://github.com/pytorch/vision 'densenet.py' (BSD-3-Clause) with
fixed kwargs passthrough and addition of dynamic global avg/max pool.
"""
import re
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.densenet import *


class DenseNet(DenseNet):
    def __init__(
            self,
            growth_rate,
            block_config,
            num_init_features,
            **kwargs
    ) -> None:

        super(DenseNet, self).__init__(**kwargs)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        
        return out, features


def _densenet(growth_rate,
              block_config,
              num_init_features,
              weights,
              progress,
              **kwargs) -> DenseNet:
    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)
    return model


def densenet121(*, weights=None, progress: bool = True, **kwargs):
    return _densenet(32, (6, 12, 24, 16), 64, weights, progress, **kwargs)
