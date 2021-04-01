import modules
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, model_size, latent_size):
        super().__init__()

        self.conv = nn.Sequential(
            modules.ConvNorm2d(1, model_size, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            modules.ConvNorm2d(model_size, model_size * 2, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            modules.ConvNorm2d(model_size * 2, model_size * 4, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(model_size * 4, latent_size * 2, 7))

    def forward(self, input):
        input = self.conv(input)
        input = input.view(*input.size()[:2])
        mean, log_var = torch.split(input, input.size(-1) // 2, dim=-1)

        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, model_size, latent_size):
        super().__init__()

        self.conv = nn.Sequential(
            modules.ConvTransposeNorm2d(latent_size, model_size * 4, 7),
            nn.LeakyReLU(0.2, inplace=True),

            modules.ConvTransposeNorm2d(model_size * 4, model_size * 2, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            modules.ConvTransposeNorm2d(model_size * 2, model_size, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(model_size, 1, 3, padding=1),
            nn.Tanh())

    def forward(self, input):
        input = input.view(*input.size(), 1, 1)
        input = self.conv(input)

        return input
