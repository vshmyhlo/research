import math

import torch
from torch import nn as nn
from torch.nn import functional as F


class LeakyReLU(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.scale = math.sqrt(2)

    def forward(self, input):
        return self.relu(input) * self.scale


class WeightParameter(nn.Module):
    def __init__(
        self,
        shape: tuple,
        lr_mul: float = 1.0,
    ):
        super().__init__()

        if len(shape) == 2:
            out_f, in_f = shape
            he_std = 1 / math.sqrt(in_f)
        elif len(shape) == 4:
            out_f, in_f, k_h, k_w = shape
            he_std = 1 / math.sqrt(in_f * k_h * k_w)
        else:
            raise ValueError(f"invalid shape {shape}")

        self.param = nn.Parameter(torch.empty(*shape))
        self.scale = he_std * lr_mul
        self.init(1.0 / lr_mul)

    def init(self, std: float):
        nn.init.normal_(self.param, 0, std)

    def forward(self):
        return self.param * self.scale


class BiasParameter(nn.Module):
    def __init__(
        self,
        shape: tuple,
        init_value: float = 0.0,
        lr_mul: float = 1.0,
    ):
        super().__init__()

        self.param = nn.Parameter(torch.empty(*shape))
        self.scale = lr_mul
        self.init(init_value)

    def init(self, value: float):
        nn.init.constant_(self.param, value)

    def forward(self):
        return self.param * self.scale


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lr_mul: float = 1.0,
    ):
        super().__init__()

        self.weight = WeightParameter((out_features, in_features), lr_mul=lr_mul)

    def forward(self, input):
        return F.linear(input, self.weight())


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        lr_mul: float = 1.0,
    ):
        super().__init__()

        self.stride = stride
        self.padding = padding

        self.weight = WeightParameter(
            (out_channels, in_channels, kernel_size, kernel_size), lr_mul=lr_mul
        )

    def forward(self, input):
        return F.conv2d(input, self.weight(), stride=self.stride, padding=self.padding)


class ModConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        lr_mul: float = 1.0,
        demodulate: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()

        self.stride = stride
        self.padding = padding
        self.demodulate = demodulate
        self.eps = eps

        self.affine = nn.Sequential(
            Linear(style_channels, in_channels, lr_mul=lr_mul),
            Bias(in_channels, init_value=1.0),
        )
        self.weight = WeightParameter(
            (out_channels, in_channels, kernel_size, kernel_size), lr_mul=lr_mul
        )

    def forward(self, input, style):
        b, _, h, w = input.size()

        # get weight
        weight = self.weight()
        out_c, in_c, _, _ = weight.size()

        # modulate
        style = self.affine(style)
        w1 = style[:, None, :, None, None]
        w2 = weight[None, :, :, :, :]
        weight = w1 * w2

        # demodulate
        if self.demodulate:
            demod = torch.rsqrt(weight.square().sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weight = weight * demod

        # convolution
        input = input.reshape(1, b * in_c, h, w)
        _, _, *ws = weight.shape
        weight = weight.reshape(b * out_c, *ws)

        input = F.conv2d(input, weight, stride=self.stride, padding=self.padding, groups=b)
        input = input.reshape(b, out_c, h, w)

        return input


class NoiseBroadcast(nn.Module):
    def __init__(self, num_channels=1):
        super().__init__()

        self.scale = nn.Parameter(torch.empty(1, num_channels, 1, 1))
        self.init()

    def init(self):
        nn.init.zeros_(self.scale)

    def forward(self, input):
        b, _, h, w = input.size()
        noise = torch.empty(b, 1, h, w, dtype=input.dtype, device=input.device).normal_(0, 1)
        input = input + noise * self.scale

        return input


class Bias(nn.Module):
    def __init__(self, num_features, init_value: float = 0.0, lr_mul: float = 1.0):
        super().__init__()

        self.bias = BiasParameter((1, num_features), init_value=init_value, lr_mul=lr_mul)

    def forward(self, input):
        return input + self.bias()


class Bias2d(nn.Module):
    def __init__(self, num_channels, init_value: float = 0.0, lr_mul: float = 1.0):
        super().__init__()

        self.bias = BiasParameter((1, num_channels, 1, 1), init_value=init_value, lr_mul=lr_mul)

    def forward(self, input):
        return input + self.bias()
