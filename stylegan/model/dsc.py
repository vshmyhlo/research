import math

import torch
from torch import nn as nn

from stylegan.model.modules import Bias, Bias2d, Conv2d, LeakyReLU, Linear


class Dsc(nn.Module):
    def __init__(self, image_size, base_channels, max_channels):
        super().__init__()

        self.from_rgb = FromRGB(base_channels)

        channels = [
            (
                min(base_channels * 2 ** i, max_channels),
                min(base_channels * 2 ** (i + 1), max_channels),
            )
            for i in range(round(math.log2(image_size / 4)))
        ]
        print("Dsc", *channels, sep="\n")

        blocks = [Block(in_channels, out_channels) for in_channels, out_channels in channels]
        self.blocks = nn.Sequential(*blocks)
        self.output = Output(channels[-1][1], channels[-1][1])

    def forward(self, image):
        input = self.from_rgb(image)
        input = self.blocks(input)
        input = self.output(input)

        return input


class FromRGB(nn.Sequential):
    def __init__(self, out_channels):
        super().__init__(
            Conv2d(3, out_channels, kernel_size=1),
            Bias2d(out_channels),
            LeakyReLU(),
        )


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.residual = nn.Sequential(
            Conv2d(in_channels, out_channels, kernel_size=1),
            nn.UpsamplingBilinear2d(scale_factor=0.5),
        )
        self.conv1 = nn.Sequential(
            Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            Bias2d(in_channels),
            LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.UpsamplingBilinear2d(scale_factor=0.5),
            Bias2d(out_channels),
            LeakyReLU(),
        )

    def forward(self, input):
        residual = self.residual(input)
        input = self.conv1(input)
        input = self.conv2(input)
        input = (input + residual) * (1 / math.sqrt(2))

        return input


class Output(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super().__init__()

        self.batch_std = BatchSTD(group_size=4, num_channels=1)
        self.conv = nn.Sequential(
            Conv2d(in_channels + 1, mid_channels, kernel_size=3, padding=1),
            Bias2d(mid_channels),
            LeakyReLU(),
        )
        self.linear = nn.Sequential(
            Linear(mid_channels * 16, mid_channels),
            Bias(mid_channels),
            LeakyReLU(),
        )
        self.output = nn.Sequential(
            Linear(mid_channels, 1),
            Bias(1),
        )

    def forward(self, input):
        input = self.batch_std(input)
        input = self.conv(input)
        b, c, h, w = input.size()
        input = input.view(b, c * h * w)
        input = self.linear(input)
        input = self.output(input)
        input = input.view(input.size(0))

        return input


class BatchSTD(nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, input):
        b, c, h, w = input.shape
        g = (
            torch.min(torch.as_tensor(self.group_size), torch.as_tensor(b))
            if self.group_size is not None
            else b
        )
        f = self.num_channels
        c = c // f

        stat = input.reshape(
            g, -1, f, c, h, w
        )  # [GnFcHW] Split minibatch b into n groups of size g, and channels c into f groups of size c.
        stat = stat - stat.mean(dim=0)  # [GnFcHW] Subtract mean over group.
        stat = stat.square().mean(dim=0)  # [nFcHW]  Calc variance over group.
        stat = (stat + 1e-8).sqrt()  # [nFcHW]  Calc stddev over group.
        stat = stat.mean(dim=[2, 3, 4])  # [nF]     Take average over channels and pixels.
        stat = stat.reshape(-1, f, 1, 1)  # [nF11]   Add missing dimensions.

        # stat is [gs, c, 1, 1]
        stat = stat.repeat(g, 1, h, w)  # [NFHW]   Replicate over group and pixels.
        input = torch.cat([input, stat], dim=1)  # [NCHW]   Append to input as new channels.

        return input
