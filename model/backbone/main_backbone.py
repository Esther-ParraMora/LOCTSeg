# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

import torch.nn as nn
from typing import Any
from torch.nn.functional import pad
from torch import cat, div

__all__ = ['mainBackboneLOCTSeg']


class FireBlockDepthWise(nn.Module):

    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super(FireBlockDepthWise, self).__init__()
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_bn = nn.BatchNorm2d(squeeze_planes)
        self.squeeze_relu = nn.ReLU()

        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_bn = nn.BatchNorm2d(expand1x1_planes)
        self.expand1x1_relu = nn.ReLU()
        self.expand3x3 = nn.Sequential(
            nn.Conv2d(in_channels=squeeze_planes, out_channels=squeeze_planes, kernel_size=3, groups=squeeze_planes,
                      padding=1),
            nn.Conv2d(in_channels=squeeze_planes, out_channels=expand3x3_planes, kernel_size=1)
        )
        self.expand3x3_bn = nn.BatchNorm2d(expand3x3_planes)
        self.expand3x3_relu = nn.ReLU()

    def forward(self, x):
        x = self.squeeze_relu(self.squeeze_bn(self.squeeze(x)))
        return cat([
            self.expand1x1_relu(self.expand1x1_bn(self.expand1x1(x))),
            self.expand3x3_relu(self.expand3x3_bn(self.expand3x3(x)))
        ], 1)

class DecovBlock(nn.Module):
    def __init__(self):
        super(DecovBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.deconv_bn = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.deconv_bn(self.deconv(x)))


class Net(nn.Module):

    def __init__(self, input_channels):
        super(Net, self).__init__()
        self.init_layers = nn.Sequential(
            nn.Conv2d(input_channels, 96, kernel_size=7, stride=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        )
        self.Fire1 = FireBlockDepthWise(96, 16, 64, 64)
        self.Fire2 = FireBlockDepthWise(128, 16, 64, 64)
        self.Fire3 = FireBlockDepthWise(128, 32, 128, 128)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.Fire4 = FireBlockDepthWise(256, 32, 128, 128)
        self.deconv = DecovBlock()
        self.conv1x1 = nn.Conv2d(256, 128, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.init_layers(x)
        x = x1 = self.Fire1(x)
        x = self.Fire2(x)
        x = self.Fire3(x)
        x = self.maxpool1(x)
        x = self.Fire4(x)
        x = self.deconv(x)
        # diffY = x1.size()[2] - x.size()[2]
        # diffX = x1.size()[3] - x.size()[3]
        # x = pad(x, [div(diffX, 2, rounding_mode='floor'),
        #             diffX - div(diffX, 2, rounding_mode='floor'),
        #             div(diffY, 2, rounding_mode='floor'),
        #             diffY - div(diffY, 2, rounding_mode='floor')])
        #
        # # x = interpolate(x, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        x = cat([x, x1], dim=1)
        x = self.conv1x1(x)
        return x


def _backbone(
        input_channels: int
) -> Net:
    model = Net(input_channels=input_channels)
    return model


def mainBackboneLOCTSeg(input_channels: int = 3) -> Net:
    return _backbone(input_channels)
