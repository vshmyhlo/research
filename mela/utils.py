import torch
from torch.nn import functional as F


def drop(input):
    def standartize(input):
        return (input - input.min()) / (input.max() - input.min())

    _, h, w = input.size()

    p = input.mean(0)
    p = standartize(p)
    p = p**2
    p = standartize(p)

    p = p.view(1, 1, h, w)
    p = F.upsample(p, scale_factor=1 / 32, mode='bilinear')
    m = (torch.rand_like(p) > p / 8).float()
    m = F.upsample(m, scale_factor=32, mode='nearest')
    m = m.view(1, h, w)

    input *= m

    return input
