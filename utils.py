import math

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt

COLORS = ('#1f77b4', '#ff7f0e', '#3ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf')


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
