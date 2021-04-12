import math
import random

import torch
import torch.nn.functional as F
from torch import nn as nn

from stylegan.model.modules import Bias, Bias2d, LeakyReLU, Linear, ModConv2d, NoiseBroadcast


class Gen(nn.Module):
    def __init__(self, image_size, base_channels, max_channels, z_channels):
        super().__init__()

        self.mapping = MappingNetwork(z_channels, lr_mul=0.01)
        self.style_mixing = StyleMixing(0.9)

        channels = [
            (
                min(base_channels * 2 ** (i + 1), max_channels),
                min(base_channels * 2 ** i, max_channels),
            )
            for i in reversed(range(round(math.log2(image_size / 4))))
        ]
        print("Gen", *channels, sep="\n")
        self.num_layers = len(channels) * 3 + 2
        print("Gen num_layers {}".format(self.num_layers))

        self.const = nn.Parameter(torch.empty(1, channels[0][0], 4, 4))
        self.input = nn.ModuleDict(
            {
                "conv": BlockLayer(channels[0][0], channels[0][0], z_channels),
                "to_rgb": ToRGB(channels[0][0], z_channels),
            }
        )
        blocks = [
            nn.ModuleDict(
                {
                    "conv": Block(in_channels, out_channels, z_channels),
                    "to_rgb": ToRGB(out_channels, z_channels),
                }
            )
            for in_channels, out_channels in channels
        ]
        self.blocks = nn.ModuleList(blocks)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.init()

    def init(self):
        nn.init.normal_(self.const, 0, 1)

    def forward(self, z1, z2=None, mix_cutoff=None):
        def z_to_w(input):
            input = self.mapping(input)
            input = layer_broadcast(input, self.num_layers)
            return input

        batch_size = z1.size(0)
        w1 = z_to_w(z1)

        if self.training:
            assert z2 is not None
            w2 = z_to_w(z2)
            w = self.style_mixing(w1, w2)
        elif z2 is not None:
            w2 = z_to_w(z2)
            w = self.style_mixing(w1, w2, cutoff=mix_cutoff)
        else:
            w = w1

        w_stack = w
        w = list(w.unbind(0))
        assert len(w) == self.num_layers

        input = self.const.repeat(batch_size, 1, 1, 1)
        input = self.input.conv(input, w.pop(0))
        image = self.input.to_rgb(input, w.pop(0))

        for block in self.blocks:
            input = block.conv(input, w)
            image = self.upsample(image) + block.to_rgb(input, w.pop(0))

        assert len(w) == 0

        return image, w_stack


class ToRGB(nn.Module):
    def __init__(self, in_channels, style_channels):
        super().__init__()

        self.conv = ModConv2d(
            in_channels,
            3,
            style_channels,
            kernel_size=1,
            demodulate=False,
        )
        self.bias = Bias2d(3)

    def forward(self, input, w):
        input = self.conv(input, w)
        input = self.bias(input)

        return input


class MappingNetwork(nn.Module):
    def __init__(self, num_features, lr_mul=1.0):
        super().__init__()

        layers = [
            nn.Sequential(
                Linear(num_features, num_features, lr_mul=lr_mul),
                Bias(num_features, lr_mul=lr_mul),
                LeakyReLU(),
            )
            for _ in range(8)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        input = F.normalize(input, p=2, dim=1)
        input = self.layers(input)

        return input


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels):
        super().__init__()

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.layer1 = BlockLayer(in_channels, out_channels, style_channels)
        self.layer2 = BlockLayer(out_channels, out_channels, style_channels)

    def forward(self, input, w):
        input = self.upsample(input)
        input = self.layer1(input, w.pop(0))
        input = self.layer2(input, w.pop(0))

        return input


class BlockLayer(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels):
        super().__init__()

        self.conv = ModConv2d(in_channels, out_channels, style_channels, kernel_size=3, padding=1)
        self.noise = NoiseBroadcast()
        self.bias = Bias2d(out_channels)
        self.relu = LeakyReLU()

    def forward(self, input, w):
        input = self.conv(input, w)
        input = self.noise(input)
        input = self.bias(input)
        input = self.relu(input)

        return input


class StyleMixing(nn.Module):
    def __init__(self, prob):
        super().__init__()

        self.prob = prob

    def forward(self, w1, w2, cutoff=None):
        assert w1.size() == w2.size()
        l, b, _ = w1.size()

        l_index = torch.arange(0, l, device=w1.device).view(l, 1, 1)

        if self.training:
            assert cutoff is None
            cutoff = torch.randint(1, l, size=(1, b, 1), device=w1.device)
            mask = (l_index < cutoff) | (torch.rand(1, b, 1) > self.prob)
        else:
            mask = l_index < cutoff

        mix = torch.where(mask, w1, w2)

        return mix


def layer_broadcast(input, num_layers):
    return input.unsqueeze(0).repeat(num_layers, 1, 1)
