import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import _calculate_correct_fan, calculate_gain

# TODO: check for conv sizes
EPS = 1e-8


class NoOp(nn.Sequential):
    def __init__(self):
        super().__init__()


class AdditiveNoise(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.weight = nn.Parameter(torch.Tensor(1, in_channels, 1, 1))

        nn.init.constant_(self.weight, 0.)

    def forward(self, input):
        noise = torch.normal(0., 1., size=(1, 1, input.size(2), input.size(3)), device=input.device)

        return input + noise * self.weight


class ReLU(nn.LeakyReLU):
    def __init__(self):
        super().__init__(0.2)


class Conv(nn.Conv2d):
    def forward(self, input):
        weight = kaiming_normal_scale(self.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class ConvTranspose(nn.ConvTranspose2d):
    def forward(self, input, output_size=None):
        weight = kaiming_normal_scale(self.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size)

        return F.conv_transpose2d(
            input, weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)


class PixelNorm(nn.Module):
    def __init__(self, eps=EPS):
        super().__init__()

        self.eps = eps

    def forward(self, input):
        norm = torch.sqrt((input**2).mean(1, keepdim=True) + self.eps)
        input = input / norm

        return input


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, channels, latent_size, eps=EPS):
        super().__init__()

        self.mean_std = nn.Conv2d(latent_size, channels * 2, 1)
        self.eps = eps

    def forward(self, input, latent):
        input = (input - input.mean((2, 3), keepdim=True)) / (input.std((2, 3), keepdim=True) + self.eps)
        mean, std = self.mean_std(latent).split(input.size(1), dim=1)
        input = (input * (std + 1)) + mean

        return input


class Upsample(nn.UpsamplingBilinear2d):
    pass


class Generator(nn.Module):
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

            self.conv = ConvTranspose(in_channels, out_channels, 4)
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
            self.conv = Conv(in_channels, out_channels, 3, padding=1)
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

    def __init__(self, image_size, latent_size):
        def build_level_layers(level, base_channels=16):
            in_channels = base_channels * 2**(max_level - level)
            out_channels = base_channels * 2**(max_level - level - 1)

            if level == 0:
                conv = self.SeqBlock(
                    self.ZeroBlock(latent_size, out_channels, latent_size),
                    self.MidBlock(out_channels, out_channels, latent_size))
            else:
                conv = self.SeqBlock(
                    self.MidBlock(in_channels, out_channels, latent_size, upsample=True),
                    self.MidBlock(out_channels, out_channels, latent_size))

            to_rgb = nn.Sequential(
                Conv(out_channels, 3, 1),
                nn.Tanh())

            return nn.ModuleDict(OrderedDict({
                'conv': conv,
                'to_rgb': to_rgb,
            }))

        super().__init__()

        self.upsample = Upsample(scale_factor=2)
        max_level = np.log2(image_size - 1).astype(np.int32)
        levels = np.arange(max_level)
        self.blocks = nn.ModuleDict(OrderedDict({
            str(level): build_level_layers(level)
            for level in levels
        }))

    def forward(self, input, level, a):
        input = input.view(input.size(0), input.size(1), 1, 1)
        latent = input

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


class Discriminator(nn.Module):
    class ZeroBlock(nn.Sequential):
        def __init__(self, in_channels):
            super().__init__(
                Conv(in_channels, 1, 4))

    class MidBlock(nn.Sequential):
        def __init__(self, in_channels, out_channels, downsample=False):
            super().__init__(
                Conv(in_channels, out_channels, 3, padding=1),
                ReLU(),
                Upsample(scale_factor=0.5) if downsample else NoOp())

    def __init__(self, image_size):
        def build_level_layers(level, base_channels=16):
            in_channels = base_channels * 2**(max_pow - level - 1)
            out_channels = base_channels * 2**(max_pow - level)

            if level == 0:
                conv = nn.Sequential(
                    self.MidBlock(in_channels, in_channels),
                    self.ZeroBlock(in_channels))
            else:
                conv = nn.Sequential(
                    self.MidBlock(in_channels, in_channels),
                    self.MidBlock(in_channels, out_channels, downsample=True))

            from_rgb = nn.Sequential(
                Conv(3, in_channels, 1),
                ReLU())

            return nn.ModuleDict(OrderedDict({
                'conv': conv,
                'from_rgb': from_rgb,
            }))

        super().__init__()

        self.downsample = Upsample(scale_factor=0.5)
        max_pow = np.log2(image_size - 1).astype(np.int32)
        levels = np.arange(max_pow)
        self.blocks = nn.ModuleDict(OrderedDict({
            str(level): build_level_layers(level)
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
