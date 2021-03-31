import torch.nn as nn


class Gen(nn.Module):
    def __init__(self, num_channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            #
            nn.ConvTranspose2d(512, 256, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            #
            nn.ConvTranspose2d(256, 128, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            #
            nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            #
            nn.ConvTranspose2d(64, num_channels, 5, stride=2, padding=2, output_padding=1),
            nn.Tanh(),
        )

        self.apply(weights_init)

    def forward(self, input):
        b, c = input.size()
        input = input.view(b, c, 1, 1)
        input = self.net(input)

        return input


class Dsc(nn.Module):
    def __init__(self, num_channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(num_channels, 64, 5, stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            #
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            #
            nn.Conv2d(128, 256, 5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            #
            nn.Conv2d(256, 512, 5, stride=2, padding=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            #
            nn.Conv2d(512, 1, 4),
        )

        self.apply(weights_init)

    def forward(self, input):
        input = self.net(input)
        input = input.view(input.size(0))

        return input


def weights_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
