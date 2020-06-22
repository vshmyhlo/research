from torch import nn as nn

from fcos.model.fpn import UpsampleMerge
from fcos.model.modules import ConvNorm, ReLU


class FPN(nn.Module):
    def __init__(self, anchor_levels, featuremap_depths):
        super().__init__()

        self.c5_to_p6 = ConvNorm(featuremap_depths[5], 256, 3, stride=2)
        self.p6_to_p7 = nn.Sequential(
            ReLU(),
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
