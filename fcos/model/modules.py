import torch
from torch import nn as nn


class ReLU(nn.ReLU):
    def __init__(self):
        super().__init__(inplace=True)


class Norm(nn.GroupNorm):
    def __init__(self, num_features):
        super().__init__(num_channels=num_features, num_groups=32)


class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, init=None):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

        if init is not None:
            init.init_(self.weight)
        if bias:
            nn.init.constant_(self.bias, 0.)


class ConvNorm(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, init=None):
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


class BatchNormFreeze(nn.Module):
    def __init__(self, module):
        def freeze_bn(m):
            if isinstance(m, nn.BatchNorm2d):
                for p in m.parameters():
                    p.requires_grad = False

        super().__init__()

        self.module = module
        self.apply(freeze_bn)

    def forward(self, *input):
        return self.module(*input)

    def train(self, mode=True):
        def freeze_bn(m):
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        super().train(mode)

        self.apply(freeze_bn)
