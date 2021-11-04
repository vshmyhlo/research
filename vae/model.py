import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, num_channels, model_size, latent_size):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(num_channels, model_size, 3, padding=1, bias=False),
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
            nn.Conv2d(model_size * 4, model_size * 8, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(model_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            #
            nn.Conv2d(model_size * 8, latent_size * 2, 4),
        )

    def forward(self, input):
        input = self.conv(input)
        input = input.view(input.size(0), input.size(1))

        mean, log_var = torch.chunk(input, 2, dim=1)
        dist = torch.distributions.Normal(mean, (log_var / 2).exp())

        return dist


class Decoder(nn.Module):
    def __init__(self, num_channels, model_size, latent_size):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(latent_size, model_size * 8, 4),
            nn.BatchNorm2d(model_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            #
            nn.ConvTranspose2d(model_size * 8, model_size * 4, 4, stride=2, padding=1, bias=False),
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
            nn.Conv2d(model_size, num_channels * 2, 3, padding=1),
        )

        self.log_var = nn.Parameter(torch.empty(()))
        nn.init.constant_(self.log_var, 0)

    def forward(self, input):
        input = input.view(*input.size(), 1, 1)
        input = self.conv(input)

        mean, log_var = torch.chunk(input, 2, dim=1)
        dist = torch.distributions.Normal(mean, (log_var / 2).exp())

        # mean, _ = torch.chunk(input, 2, dim=1)
        # dist = torch.distributions.Normal(mean, (self.log_var / 2).exp())

        return dist


class Model(nn.Module):
    def __init__(self, num_channels, model_size, latent_size):
        super().__init__()

        self.encoder = Encoder(num_channels, model_size, latent_size)
        self.decoder = Decoder(num_channels, model_size, latent_size)
