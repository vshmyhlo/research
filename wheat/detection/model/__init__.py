import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from detection.anchor_utils import flatten_detection_map
from detection.model.backbone import EfficientNetB0, ResNet50


class ReLU(nn.ReLU):
    pass


class Norm(nn.GroupNorm):
    def __init__(self, num_features):
        super().__init__(num_channels=num_features, num_groups=32)


class ConvNorm(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2),
            Norm(out_channels))


class UpsampleMerge(nn.Module):
    def __init__(self, c_channels):
        super().__init__()

        self.projection = ConvNorm(c_channels, 256, 1)
        self.output = ConvNorm(256, 256, 3)

    def forward(self, p, c):
        # TODO: assert sizes

        p = F.interpolate(p, size=(c.size(2), c.size(3)), mode='nearest')
        c = self.projection(c)
        input = p + c
        input = self.output(input)

        return input


# TODO: optimize level calculation
class FPN(nn.Module):
    def __init__(self, anchor_levels, featuremap_depths):
        super().__init__()

        self.c5_to_p6 = ConvNorm(featuremap_depths[5], 256, 3, stride=2)
        self.p6_to_p7 = nn.Sequential(
            ReLU(inplace=True),
            ConvNorm(256, 256, 3, stride=2)) if anchor_levels[7] else None
        self.c5_to_p5 = ConvNorm(featuremap_depths[5], 256, 1)
        self.p5c4_to_p4 = UpsampleMerge(featuremap_depths[4])
        self.p4c3_to_p3 = UpsampleMerge(featuremap_depths[3])
        self.p3c2_to_p2 = UpsampleMerge(featuremap_depths[2]) if anchor_levels[2] else None

    def forward(self, input):
        p6 = self.c5_to_p6(input[5])
        p7 = self.p6_to_p7(p6) if self.p6_to_p7 is not None else None
        p5 = self.c5_to_p5(input[5])
        p4 = self.p5c4_to_p4(p5, input[4])
        p3 = self.p4c3_to_p3(p4, input[3])
        p2 = self.p3c2_to_p2(p3, input[2]) if self.p3c2_to_p2 is not None else None

        input = [None, None, p2, p3, p4, p5, p6, p7]

        return input


class HeadSubnet(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            ConvNorm(in_channels, in_channels, 3),
            ReLU(inplace=True),
            ConvNorm(in_channels, in_channels, 3),
            ReLU(inplace=True),
            ConvNorm(in_channels, in_channels, 3),
            ReLU(inplace=True),
            ConvNorm(in_channels, in_channels, 3),
            ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1))


class FlattenDetectionMap(nn.Module):
    def __init__(self, num_anchors):
        super().__init__()

        self.num_anchors = num_anchors

    def forward(self, input):
        return flatten_detection_map(input, self.num_anchors)


class RetinaNet(nn.Module):
    def __init__(self, model, num_classes, anchors_per_level, anchor_levels, freeze_bn=False):
        super().__init__()

        self.freeze_bn = freeze_bn

        if model.backbone == 'resnet50':
            self.backbone = ResNet50()
        elif model.backbone == 'effnetb0':
            self.backbone = EfficientNetB0()
        else:
            raise AssertionError('invalid model.backbone'.format(model.backbone))

        self.fpn = FPN(anchor_levels, self.backbone.featuremap_depths)
        self.class_head = HeadSubnet(256, anchors_per_level * num_classes)
        self.loc_head = HeadSubnet(256, anchors_per_level * 4)
        self.flatten = FlattenDetectionMap(anchors_per_level)

        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                for p in m.parameters():
                    p.requires_grad = False

        modules = itertools.chain(
            self.fpn.modules(),
            self.class_head.modules(),
            self.loc_head.modules())
        for m in modules:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0., std=0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        pi = 0.01
        nn.init.constant_(self.class_head[-1].bias, -math.log((1 - pi) / pi))

    def forward(self, input):
        backbone_output = self.backbone(input)
        fpn_output = self.fpn(backbone_output)

        class_output = torch.cat([self.flatten(self.class_head(x)) for x in fpn_output if x is not None], 1)
        loc_output = torch.cat([self.flatten(self.loc_head(x)) for x in fpn_output if x is not None], 1)

        return class_output, loc_output

    def train(self, mode=True):
        super().train(mode)

        if not self.freeze_bn:
            return

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
