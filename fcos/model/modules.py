import torch
from torch import nn as nn


class ReLU(nn.ReLU):
    pass


class Norm(nn.GroupNorm):
    def __init__(self, num_features):
        super().__init__(num_channels=num_features, num_groups=32)


class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, init='linear'):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

        nn.init.kaiming_normal_(self.weight, nonlinearity=init)
        if bias:
            nn.init.zeros_(self.bias)


class ConvNorm(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, init='linear'):
        super().__init__(
            Conv(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False, init=init),
            Norm(out_channels))


class Scale(nn.Module):
    def __init__(self):
        super().__init__()

        self.scale = nn.Parameter(torch.empty((), dtype=torch.float))

        nn.init.constant_(self.scale, 1.)

    def forward(self, input):
        return input * self.scale
