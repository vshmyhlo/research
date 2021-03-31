from torch import nn as nn

from cycle_gan.model.modules import LeakyReLU


class Dsc(nn.Module):
    def __init__(self):
        super().__init__()

        self.input = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            LeakyReLU(),
        )
        self.mid = nn.Sequential(
            Block(16, 32),
            Block(32, 64),
            Block(64, 128),
            Block(128, 256),
            Block(256, 512),
        )
        self.output = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1),
        )

        self.apply(self.init)

    def init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, input):
        input = self.input(input)
        input = self.mid(input)
        input = self.output(input)

        return input


class Block(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=0.5),
            nn.BatchNorm2d(out_channels),
            LeakyReLU(),
        )
