import math

import torch
import torch.nn as nn


def vector_to_image(input):
    b, c = input.size()
    return input.view(b, c, 1, 1)


class AdaIN(nn.Module):
    def __init__(self, num_channels, w_channels):
        super().__init__()

        self.mean = nn.Linear(w_channels, num_channels)
        self.std = nn.Linear(w_channels, num_channels)

        nn.init.zeros_(self.mean.bias)
        nn.init.ones_(self.std.bias)

    def forward(self, input, w):
        mean = input.mean((2, 3), keepdim=True)
        std = input.std((2, 3), keepdim=True)
        input = (input - mean) / (std + 1e-7)

        mean = vector_to_image(self.mean(w))
        std = vector_to_image(self.std(w))
        input = input * std + mean

        return input


class GenBlockInit(nn.Module):
    def __init__(self, in_channels, out_channels, w_channels):
        super().__init__()

        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, bias=False)
        self.norm1 = AdaIN(out_channels, w_channels)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, input, w):
        input = self.conv1(input)
        input = self.norm1(input, w)
        input = self.relu1(input)

        return input


class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, w_channels):
        super().__init__()

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = AdaIN(out_channels, w_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = AdaIN(out_channels, w_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, input, w):
        input = self.upsample(input)

        input = self.conv1(input)
        input = self.norm1(input, w)
        input = self.relu1(input)

        input = self.conv2(input)
        input = self.norm2(input, w)
        input = self.relu2(input)

        return input


class Gen(nn.Module):
    def __init__(self, image_size, image_channels, base_channels):
        super().__init__()

        w_channels = base_channels
        self.mapping = nn.Sequential(
            nn.Linear(100, base_channels),
            nn.ReLU(inplace=True),
            nn.Linear(base_channels, base_channels),
            nn.ReLU(inplace=True),
        )
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        blocks = [
            nn.ModuleDict(
                {
                    "conv": GenBlockInit(base_channels, base_channels, w_channels),
                    "rgb": nn.Sequential(
                        nn.Conv2d(
                            base_channels,
                            image_channels,
                            kernel_size=3,
                            padding=1,
                        ),
                    ),
                }
            )
        ]
        for _ in range(round(math.log2(image_size / 4))):
            blocks.append(
                nn.ModuleDict(
                    {
                        "conv": GenBlock(base_channels, base_channels // 2, w_channels),
                        "rgb": nn.Sequential(
                            nn.Conv2d(
                                base_channels // 2,
                                image_channels,
                                kernel_size=3,
                                padding=1,
                            ),
                        ),
                    }
                )
            )
            base_channels //= 2

        base_channels *= 2
        print(base_channels)
        assert base_channels >= 8
        self.blocks = nn.ModuleList(blocks)
        self.apply(weights_init)

    def forward(self, input):
        w = self.mapping(input)
        input = vector_to_image(w)

        rgb = torch.zeros(input.size(0), 3, 2, 2, dtype=input.dtype, device=input.device)
        for block in self.blocks:
            input = block["conv"](input, w)
            rgb = self.upsample(rgb) + block["rgb"](input)

        return rgb


class Dsc(nn.Module):
    def __init__(self, image_size, image_channels, base_channels, kernel_size):
        super().__init__()

        layers = [
            nn.Sequential(
                nn.Conv2d(base_channels, 1, kernel_size=4),
            )
        ]

        for _ in range(round(math.log2(image_size / 4) - 1)):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        base_channels // 2,
                        base_channels,
                        kernel_size=kernel_size,
                        stride=2,
                        padding=kernel_size // 2,
                    ),
                    nn.BatchNorm2d(base_channels),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                )
            )
            base_channels //= 2

        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    image_channels,
                    base_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=kernel_size // 2,
                ),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )
        )

        print(base_channels)
        assert base_channels >= 8
        self.net = nn.Sequential(*reversed(layers))
        self.apply(weights_init)

    def forward(self, input):
        input = self.net(input)
        input = input.view(input.size(0))

        return input


def weights_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
