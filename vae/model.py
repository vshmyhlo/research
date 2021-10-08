import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, model_size, latent_size):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, model_size, 3, padding=1, bias=False),
            nn.BatchNorm2d(model_size),
            nn.LeakyReLU(0.2, inplace=True),
            #
            nn.Conv2d(model_size, model_size * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(model_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            #
            nn.Conv2d(model_size * 2, model_size * 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(model_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            #
            nn.Conv2d(model_size * 4, latent_size * 2, 7),
        )

    def forward(self, input):
        input = self.conv(input)
        input = input.view(input.size(0), input.size(1))

        mean, log_std = torch.chunk(input, 2, dim=1)
        dist = torch.distributions.Normal(mean, log_std.exp())

        return dist


class Decoder(nn.Module):
    def __init__(self, model_size, latent_size):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(latent_size, model_size * 4, 7),
            nn.BatchNorm2d(model_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            #
            nn.ConvTranspose2d(model_size * 4, model_size * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(model_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            #
            nn.ConvTranspose2d(model_size * 2, model_size, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(model_size),
            nn.LeakyReLU(0.2, inplace=True),
            #
            nn.Conv2d(model_size, 2, 3, padding=1),
        )

        # self.log_std = nn.Parameter(torch.empty(()))
        # nn.init.constant_(self.log_std, 0)

    def forward(self, input):
        input = input.view(*input.size(), 1, 1)
        input = self.conv(input)

        mean, log_std = torch.chunk(input, 2, dim=1)
        dist = torch.distributions.Normal(mean, log_std.exp())

        # mean = input
        # dist = torch.distributions.Normal(mean, self.log_std.exp())

        return dist
