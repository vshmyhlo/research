import math

import numpy as np
import torch
import torch.optim
import torchvision
from matplotlib import pyplot as plt

COLORS = ('#1f77b4', '#ff7f0e', '#3ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf')


class WarmupCosineAnnealingLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, epoch_warmup, epoch_max):
        def f(epoch):
            if epoch < epoch_warmup:
                return epoch / epoch_warmup
            else:
                return (np.cos((epoch - epoch_warmup) / (epoch_max - epoch_warmup) * np.pi) + 1) / 2

        super().__init__(optimizer, f)


class Zip(object):
    def __init__(self, *iterables):
        self.iterables = iterables

    def __len__(self):
        return min(map(len, self.iterables))

    def __iter__(self):
        return zip(*self.iterables)


def compute_nrow(images):
    b, _, h, w = images.size()
    nrow = math.ceil(math.sqrt(h * b / w))

    return nrow


def make_grid(images):
    return torchvision.utils.make_grid(images, nrow=compute_nrow(images))


def bin_loss(w, sigma=0.2):
    c = 0.5 / math.sqrt(2 * math.pi * sigma**2)

    left = torch.exp(-1 * (w + 1)**2 / (2 * sigma**2))
    right = torch.exp(-1 * (w - 1)**2 / (2 * sigma**2))

    loss = -torch.log(c * left + c * right + 1e-8)

    return loss


def plot_decision_boundary(x, y, predict, colors=COLORS):
    grid = np.stack(np.meshgrid(
        np.linspace(x[:, 0].min() - 1., x[:, 0].max() + 1.),
        np.linspace(x[:, 1].min() - 1., x[:, 1].max() + 1.),
    ), -1)
    z = predict(grid.reshape((-1, 2))).reshape(grid.shape[:2])

    fig = plt.figure()
    plt.contourf(grid[:, :, 0], grid[:, :, 1], z, levels=np.arange(z.max() + 2) - 0.5, colors=colors, alpha=0.5)
    plt.scatter(x[y == 0][:, 0], x[y == 0][:, 1], s=5, c=colors[0])
    plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1], s=5, c=colors[1])

    return fig


def one_hot(input, n, dtype=torch.float):
    return torch.eye(n, dtype=dtype, device=input.device)[input]


def clip_grad_norm(grads, max_norm, norm_type=2):
    grads = list(filter(lambda grad: grad is not None, grads))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    total_norm = torch.norm(torch.stack([torch.norm(grad.detach(), norm_type) for grad in grads]), norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        grads = [grad * clip_coef for grad in grads]

    return grads


def cross_entropy(input, target, dim=-1, eps=1e-8):
    return -torch.sum(target * torch.log(input + eps), dim=dim)


def entropy(input):
    return cross_entropy(input, input)
