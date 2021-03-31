import math

import torch.nn as nn


class Gen(nn.Module):
    def __init__(self, image_size, image_channels, base_channels, kernel_size):
        super().__init__()

        layers = [
            nn.Sequential(
                nn.ConvTranspose2d(100, base_channels, kernel_size=4),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True),
            )
        ]

        for _ in range(round(math.log2(image_size / 4) - 1)):
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        base_channels,
                        base_channels // 2,
                        kernel_size=kernel_size,
                        stride=2,
                        padding=math.ceil(kernel_size / 2) - 1,
                        output_padding=kernel_size % 2,
                    ),
                    nn.BatchNorm2d(base_channels // 2),
                    nn.ReLU(inplace=True),
                )
            )
            base_channels //= 2

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    base_channels,
                    image_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=math.ceil(kernel_size / 2) - 1,
                    output_padding=kernel_size % 2,
                ),
                nn.Tanh(),
            )
        )

        print(base_channels)
        assert base_channels >= 8
        self.net = nn.Sequential(*layers)
        self.apply(weights_init)

    def forward(self, input):
        b, c = input.size()
        input = input.view(b, c, 1, 1)
        input = self.net(input)
        return input


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
