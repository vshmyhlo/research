import math
import time
from contextlib import contextmanager

import torch
from torch import nn as nn


@contextmanager
def zero_grad_and_step(opt):
    opt.zero_grad(set_to_none=True)
    yield
    opt.step()


def compute_nrow(images):
    b, _, h, w = images.size()
    nrow = math.ceil(math.sqrt(h * b / w))

    return nrow


def clip_parameters(m: nn.Module, value: float):
    if value <= 0:
        raise ValueError(f"expected clip value {value} to be > 0")

    for p in m.parameters():
        p.data.clamp_(-value, value)


def weighted_sum(a, b, w):
    assert 0 <= w <= 1
    return w * a + (1 - w) * b


def stack_images(images):
    assert isinstance(images, list)
    images = torch.stack(images, 1)
    b, nrow, c, h, w = images.size()
    images = images.view(b * nrow, c, h, w)

    return images, nrow


def validate_shape(input, shape):
    if input.dim() != len(shape):
        raise "Invalid shape {}, expected {}".format(input.size(), shape)

    for a, b in zip(input.size(), shape):
        if b is None:
            continue
        if a != b:
            raise "Invalid shape {}, expected {}".format(input.size(), shape)
