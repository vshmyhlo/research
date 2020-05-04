from collections import OrderedDict

import numpy as np
import torch.nn as nn


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d,)):
        nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity='leaky_relu')
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d,)):
        pass


class Generator(nn.Module):
    def __init__(self, image_size, latent_size):
        def build_block_for_level(level, base_channels=16):
            if level == 0:
                conv = nn.Sequential(
                    nn.ConvTranspose2d(
                        latent_size,
                        base_channels * 2**(max_level - level - 1),
                        4),
                    nn.LeakyReLU(0.1))
            else:
                conv = nn.Sequential(
                    self.upsample,
                    nn.Conv2d(
                        base_channels * 2**(max_level - level),
                        base_channels * 2**(max_level - level - 1),
                        3,
                        padding=1),
                    nn.LeakyReLU(0.1))

            to_rgb = nn.Sequential(
                nn.Conv2d(base_channels * 2**(max_level - level - 1), 3, 1),
                nn.Tanh())

            return nn.ModuleDict(OrderedDict({
                'conv': conv,
                'to_rgb': to_rgb,
            }))

        super().__init__()

        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        max_level = np.log2(image_size - 1).astype(np.int32)
        levels = np.arange(max_level)
        self.blocks = nn.ModuleDict(OrderedDict({
            str(level): build_block_for_level(level)
            for level in levels
        }))
        # print(self.blocks)
        self.apply(weights_init)

    def forward(self, input, level, a):
        input = input.view(input.size(0), input.size(1), 1, 1)

        blocks = OrderedDict()
        for k in range(level + 1):
            input = self.blocks[str(k)].conv(input)
            blocks[str(k)] = input
        del input

        right = self.blocks[str(level)].to_rgb(blocks[str(level)])
        if level == 0:
            input = right
        else:
            left = self.blocks[str(level - 1)].to_rgb(self.upsample(blocks[str(level - 1)]))
            input = (1 - a) * left + a * right

        return input


class Discriminator(nn.Module):
    def __init__(self, image_size):
        def build_block_for_level(level, base_channels=16):
            if level == 0:
                conv = nn.Sequential(
                    nn.Conv2d(
                        base_channels * 2**(max_pow - level - 1),
                        1,
                        4))
            else:
                conv = nn.Sequential(
                    nn.Conv2d(
                        base_channels * 2**(max_pow - level - 1),
                        base_channels * 2**(max_pow - level),
                        3,
                        padding=1),
                    nn.LeakyReLU(0.1),
                    self.downsample)

            from_rgb = nn.Sequential(
                nn.Conv2d(3, base_channels * 2**(max_pow - level - 1), 1),
                nn.LeakyReLU(0.1))

            return nn.ModuleDict(OrderedDict({
                'conv': conv,
                'from_rgb': from_rgb,
            }))

        super().__init__()

        self.downsample = nn.UpsamplingNearest2d(scale_factor=0.5)
        max_pow = np.log2(image_size - 1).astype(np.int32)
        levels = np.arange(max_pow)
        self.blocks = nn.ModuleDict(OrderedDict({
            str(level): build_block_for_level(level)
            for level in levels
        }))
        # print(self.blocks)
        self.apply(weights_init)

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
