from torch import nn as nn
from torch.nn import functional as F

from fcos.model.modules import ConvNorm, ReLU


class FPN(nn.Module):
    def __init__(self, featuremap_depths):
        super().__init__()

        self.c5_to_p5 = ConvNorm(featuremap_depths[5], 256, 1)

        self.p5_to_p6 = ConvNorm(256, 256, 3, stride=2, padding=1)
        self.p6_to_p7 = nn.Sequential(
            ReLU(inplace=True),
            ConvNorm(256, 256, 3, stride=2, padding=1))

        self.p5c4_to_p4 = UpsampleMerge(featuremap_depths[4])
        self.p4c3_to_p3 = UpsampleMerge(featuremap_depths[3])

    def forward(self, input):
        p5 = self.c5_to_p5(input[5])

        p6 = self.p5_to_p6(p5)
        p7 = self.p6_to_p7(p6)

        p4 = self.p5c4_to_p4(p5, input[4])
        p3 = self.p4c3_to_p3(p4, input[3])

        input = [None, None, None, p3, p4, p5, p6, p7]

        return input


class UpsampleMerge(nn.Module):
    def __init__(self, c_channels):
        super().__init__()

        self.projection = ConvNorm(c_channels, 256, 1)
        self.output = ConvNorm(256, 256, 3, padding=1)

    def forward(self, p, c):
        # TODO: assert sizes

        p = F.interpolate(p, size=(c.size(2), c.size(3)), mode='nearest')
        c = self.projection(c)
        input = p + c
        input = self.output(input)

        return input
