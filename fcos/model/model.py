import torch
import torch.nn as nn
import torch.nn.functional as F

from fcos.model.fpn import FPN
from fcos.model.modules import ReLU, ConvNorm, Conv, Scale
from fcos.utils import flatten_detection_map
from init import Normal, prior_
from object_detection.model.backbone import ResNet50


class HeadSubnet(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            ConvNorm(in_channels, in_channels, 3, padding=1, init=Normal(0, 0.01)),
            ReLU(inplace=True),
            ConvNorm(in_channels, in_channels, 3, padding=1, init=Normal(0, 0.01)),
            ReLU(inplace=True),
            ConvNorm(in_channels, in_channels, 3, padding=1, init=Normal(0, 0.01)),
            ReLU(inplace=True),
            ConvNorm(in_channels, in_channels, 3, padding=1, init=Normal(0, 0.01)),
            ReLU(inplace=True),
            Conv(in_channels, out_channels, 3, padding=1, init=Normal(0, 0.01)))


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
        self.scales = nn.ModuleList([Scale() for _ in range(5)])

        prior_(self.class_head[-1].bias, 0.01)

    def forward(self, input):
        backbone_output = self.backbone(input)
        fpn_output = self.fpn(backbone_output)

        fpn_output = [x for x in fpn_output if x is not None]
        assert len(fpn_output) == len(self.scales)

        class_output = []
        loc_output = []
        cent_output = []
        for o, scale in zip(fpn_output, self.scales):
            class_o = flatten_detection_map(self.class_head(o))
            loc_o = flatten_detection_map(self.loc_head(o))
            del o

            loc_o, cent_o = loc_o.split([4, 1], -1)
            loc_o = F.relu(scale(loc_o))
            cent_o = cent_o.squeeze(-1)

            class_output.append(class_o)
            loc_output.append(loc_o)
            cent_output.append(cent_o)

        class_output = torch.cat(class_output, 1)
        loc_output = torch.cat(loc_output, 1)
        cent_output = torch.cat(cent_output, 1)

        return class_output, loc_output, cent_output
