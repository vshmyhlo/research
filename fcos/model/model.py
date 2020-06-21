import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fcos.model.fpn import FPN
from fcos.model.modules import ReLU, ConvNorm, Conv
from fcos.utils import flatten_detection_map
from object_detection.model.backbone import ResNet50


class HeadSubnet(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            ConvNorm(in_channels, in_channels, 3, padding=1, init='relu'),
            ReLU(inplace=True),
            ConvNorm(in_channels, in_channels, 3, padding=1, init='relu'),
            ReLU(inplace=True),
            ConvNorm(in_channels, in_channels, 3, padding=1, init='relu'),
            ReLU(inplace=True),
            ConvNorm(in_channels, in_channels, 3, padding=1, init='relu'),
            ReLU(inplace=True),
            Conv(in_channels, out_channels, 3, padding=1))


class FCOS(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()

        if model.backbone == 'resnet50':
            self.backbone = ResNet50()
        else:
            raise AssertionError('invalid model.backbone'.format(model.backbone))

        self.fpn = FPN(self.backbone.featuremap_depths)
        self.class_head = HeadSubnet(256, num_classes)
        self.loc_head = HeadSubnet(256, 5)

        pi = 0.01
        nn.init.constant_(self.class_head[-1].bias, -math.log((1 - pi) / pi))

    def forward(self, input):
        backbone_output = self.backbone(input)
        fpn_output = self.fpn(backbone_output)

        class_output = torch.cat([flatten_detection_map(self.class_head(x)) for x in fpn_output if x is not None], 1)
        loc_output = torch.cat([flatten_detection_map(self.loc_head(x)) for x in fpn_output if x is not None], 1)

        loc_output, cent_output = loc_output.split([4, 1], -1)
        loc_output = F.relu(loc_output)
        cent_output = cent_output.squeeze(-1)

        return class_output, loc_output, cent_output
