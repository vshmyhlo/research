import torch
from torch import nn as nn


class LeakyReLU(nn.LeakyReLU):
    def __init__(self):
        super().__init__(0.2, inplace=True)


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
