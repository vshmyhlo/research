from collections import OrderedDict

import numpy as np
from torch import nn as nn

from gan.modules import ConvEq, MinibatchStdDev, NoOp, ReLU, Upsample


class Discriminator(nn.Module):
    def __init__(self, image_size):
        def build_level_layers(level, base_channels=16):
            in_channels = base_channels * 2 ** (max_level - level)
            out_channels = base_channels * 2 ** (max_level - level + 1)

            if level == 0:
                conv = nn.Sequential(
                    MinibatchStdDev(),
                    MidBlock(in_channels + 1, in_channels),
                    ZeroBlock(in_channels),
                )
            else:
                conv = nn.Sequential(
                    MidBlock(in_channels, in_channels),
                    MidBlock(in_channels, out_channels, downsample=True),
                )

            from_rgb = nn.Sequential(ConvEq(3, in_channels, 1), ReLU())

            return nn.ModuleDict(
                OrderedDict(
                    {
                        "conv": conv,
                        "from_rgb": from_rgb,
                    }
                )
            )

        super().__init__()

        self.downsample = Upsample(scale_factor=0.5)
        max_level = np.log2(image_size / 4).astype(np.int32)
        levels = np.arange(max_level + 1)
        self.blocks = nn.ModuleDict(
            OrderedDict({str(level): build_level_layers(level) for level in levels})
        )

    def forward(self, input, level, a):
        right = self.blocks[str(level)].conv(self.blocks[str(level)].from_rgb(input))
        if level == 0:
            input = right
        else:
            left = self.blocks[str(level - 1)].from_rgb(self.downsample(input))
            input = (1 - a) * left + a * right

        for k in range(level - 1, -1, -1):
            input = self.blocks[str(k)].conv(input)

        input = input.view(input.size(0))

        return input


class ZeroBlock(nn.Sequential):
    def __init__(self, in_channels):
        super().__init__(ConvEq(in_channels, in_channels, 4), ReLU(), ConvEq(in_channels, 1, 1))


class MidBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__(
            ConvEq(in_channels, out_channels, 3, padding=1),
            ReLU(),
            Upsample(scale_factor=0.5) if downsample else NoOp(),
        )
