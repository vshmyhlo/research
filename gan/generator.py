from collections import OrderedDict

import numpy as np
from torch import nn as nn

from gan.modules import ConvTransposeEq, AdditiveNoise, ReLU, AdaptiveInstanceNorm, Upsample, NoOp, ConvEq, PixelNorm


class Generator(nn.Module):
    def __init__(self, image_size, latent_size):
        def build_level_layers(level, base_channels=16):
            in_channels = base_channels * 2**(max_level - level)
            out_channels = base_channels * 2**(max_level - level - 1)

            if level == 0:
                conv = SeqBlock(
                    ZeroBlock(latent_size, out_channels, latent_size),
                    MidBlock(out_channels, out_channels, latent_size))
            else:
                conv = SeqBlock(
                    MidBlock(in_channels, out_channels, latent_size, upsample=True),
                    MidBlock(out_channels, out_channels, latent_size))

            to_rgb = nn.Sequential(
                ConvEq(out_channels, 3, 1),
                nn.Tanh())

            return nn.ModuleDict(OrderedDict({
                'conv': conv,
                'to_rgb': to_rgb,
            }))

        super().__init__()

        self.mapping = nn.Sequential(
            PixelNorm(),
            Mapping(latent_size))
        self.upsample = Upsample(scale_factor=2)
        max_level = np.log2(image_size - 1).astype(np.int32)
        levels = np.arange(max_level)
        self.blocks = nn.ModuleDict(OrderedDict({
            str(level): build_level_layers(level)
            for level in levels
        }))

    def forward(self, input, level, a):
        input = input.view(input.size(0), input.size(1), 1, 1)
        latent = self.mapping(input)

        blocks = OrderedDict()
        for k in range(level + 1):
            input = self.blocks[str(k)].conv(input, latent)
            blocks[str(k)] = input
        del input

        right = self.blocks[str(level)].to_rgb(blocks[str(level)])
        if level == 0:
            input = right
        else:
            left = self.blocks[str(level - 1)].to_rgb(self.upsample(blocks[str(level - 1)]))
            input = (1 - a) * left + a * right

        return input


class Mapping(nn.Sequential):
    def __init__(self, latent_size):
        super().__init__(
            ConvEq(latent_size, latent_size, 1, scale=0.01),
            ReLU(),
            ConvEq(latent_size, latent_size, 1, scale=0.01),
            ReLU(),
            ConvEq(latent_size, latent_size, 1, scale=0.01),
            ReLU())


class SeqBlock(nn.ModuleList):
    def __init__(self, *blocks):
        super().__init__(blocks)

    def forward(self, input, latent):
        for block in self:
            input = block(input, latent)

        return input


class ZeroBlock(nn.Module):
    def __init__(self, in_channels, out_channels, latent_size):
        super().__init__()

        self.conv = ConvTransposeEq(in_channels, out_channels, 4)
        self.noise = AdditiveNoise(out_channels)
        self.relu = ReLU()
        self.norm = AdaptiveInstanceNorm(out_channels, latent_size)

    def forward(self, input, latent):
        input = self.conv(input)
        input = self.noise(input)
        input = self.relu(input)
        input = self.norm(input, latent)

        return input


class MidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, latent_size, upsample=False):
        super().__init__()

        self.upsample = Upsample(scale_factor=2) if upsample else NoOp()
        self.conv = ConvEq(in_channels, out_channels, 3, padding=1)
        self.noise = AdditiveNoise(out_channels)
        self.relu = ReLU()
        self.norm = AdaptiveInstanceNorm(out_channels, latent_size)

    def forward(self, input, latent):
        input = self.upsample(input)
        input = self.conv(input)
        input = self.noise(input)
        input = self.relu(input)
        input = self.norm(input, latent)

        return input
