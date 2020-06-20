from torch import nn as nn


class ReLU(nn.ReLU):
    pass


class Norm(nn.GroupNorm):
    def __init__(self, num_features):
        super().__init__(num_channels=num_features, num_groups=32)


class ConvNorm(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2),
            Norm(out_channels))
