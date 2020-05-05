import math

import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.init import _calculate_correct_fan, calculate_gain


class NoOp(nn.Sequential):
    def __init__(self):
        super().__init__()


class Upsample(nn.UpsamplingBilinear2d):
    pass


class ReLU(nn.LeakyReLU):
    def __init__(self):
        super().__init__(0.2)


class AdditiveNoise(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.weight = nn.Parameter(torch.Tensor(1, in_channels, 1, 1))

    def forward(self, input):
        noise = torch.normal(0., 1., size=(1, 1, input.size(2), input.size(3)), device=input.device)

        return input + noise * self.weight


class PixelNorm(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()

        self.eps = eps

    def forward(self, input):
        norm = torch.sqrt((input**2).mean(1, keepdim=True) + self.eps)
        input = input / norm

        return input


class InstanceNorm(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()

        self.eps = eps

    def forward(self, input):
        mean = input.mean((2, 3), keepdim=True)
        input = input - mean
        norm = torch.sqrt((input**2).mean((2, 3), keepdim=True) + self.eps)
        input = input / norm

        return input


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, channels, latent_size):
        super().__init__()

        self.norm = InstanceNorm()
        self.style = ConvEq(latent_size, channels * 2, 1)

    def forward(self, input, latent):
        input = self.norm(input)
        mean, std = self.style(latent).split(input.size(1), dim=1)
        input = input * (std + 1) + mean
       
        return input


class ConvEq(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, scale=1.):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

        self.scale = scale

    def forward(self, input):
        weight = kaiming_normal_scale(self.weight * self.scale, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class ConvTransposeEq(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, scale=1.):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

        self.scale = scale

    def forward(self, input, output_size=None):
        weight = kaiming_normal_scale(self.weight * self.scale, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size)

        return F.conv_transpose2d(
            input, weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)


def kaiming_normal_scale(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)

    return tensor * std
