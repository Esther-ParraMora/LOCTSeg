# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 13:01:40 2022

@author: Esther
"""

from collections import OrderedDict
from typing import Optional, Dict
from torch import nn, Tensor
from torch.nn import functional as F


class _SMSimpleBackboneClassifier(nn.Module):

    def __init__(self, backbone: nn.Module, classifier: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        x = self.backbone(x)
        result = OrderedDict()
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        result["out"] = x
        return result


class _SMBackboneSkip(nn.Module):

    def __init__(self, backbone: nn.Module, classifier: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        features, x1 = self.backbone(x)
        result = OrderedDict()
        x = features
        x = self.classifier(x, x1)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        result["out"] = x
        return result
