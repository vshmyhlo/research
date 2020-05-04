import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import _calculate_correct_fan, calculate_gain


class ReLU(nn.LeakyReLU):
    def __init__(self):
        super().__init__(0.1)


class Conv(nn.Conv2d):
    def forward(self, input):
        weight = kaiming_normal_scale(self.weight, a=0.1, mode='fan_in', nonlinearity='leaky_relu')

        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class ConvTranspose(nn.ConvTranspose2d):
    def forward(self, input, output_size=None):
        weight = kaiming_normal_scale(self.weight, a=0.1, mode='fan_in', nonlinearity='leaky_relu')
        output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size)

        return F.conv_transpose2d(
            input, weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)


class PixelNorm(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()

        self.eps = eps

    def forward(self, input):
        norm = torch.sqrt((input**2).mean(1, keepdim=True) + self.eps)
        input = input / norm

        return input


class Generator(nn.Module):
    def __init__(self, image_size, latent_size):
        def build_block_for_level(level, base_channels=16):
            if level == 0:
                conv = nn.Sequential(
                    ConvTranspose(
                        latent_size,
                        base_channels * 2**(max_level - level - 1),
                        4),
                    ReLU(),
                    PixelNorm(),
                    Conv(
                        base_channels * 2**(max_level - level - 1),
                        base_channels * 2**(max_level - level - 1),
                        3,
                        padding=1),
                    ReLU(),
                    PixelNorm())
            else:
                conv = nn.Sequential(
                    self.upsample,
                    Conv(
                        base_channels * 2**(max_level - level),
                        base_channels * 2**(max_level - level - 1),
                        3,
                        padding=1),
                    ReLU(),
                    PixelNorm(),
                    Conv(
                        base_channels * 2**(max_level - level - 1),
                        base_channels * 2**(max_level - level - 1),
                        3,
                        padding=1),
                    ReLU(),
                    PixelNorm())

            to_rgb = nn.Sequential(
                Conv(base_channels * 2**(max_level - level - 1), 3, 1),
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
                    Conv(
                        base_channels * 2**(max_pow - level - 1),
                        base_channels * 2**(max_pow - level - 1),
                        3,
                        padding=1),
                    ReLU(),
                    Conv(
                        base_channels * 2**(max_pow - level - 1),
                        1,
                        4))
            else:
                conv = nn.Sequential(
                    Conv(
                        base_channels * 2**(max_pow - level - 1),
                        base_channels * 2**(max_pow - level - 1),
                        3,
                        padding=1),
                    ReLU(),
                    Conv(
                        base_channels * 2**(max_pow - level - 1),
                        base_channels * 2**(max_pow - level),
                        3,
                        padding=1),
                    ReLU(),
                    self.downsample)

            from_rgb = nn.Sequential(
                Conv(3, base_channels * 2**(max_pow - level - 1), 1),
                ReLU())

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


def kaiming_normal_scale(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)

    return tensor * std
