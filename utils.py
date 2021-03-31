import math
from contextlib import contextmanager

import numpy as np
import torch
import torch.optim
import torchvision
from matplotlib import pyplot as plt

COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#3ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)


@contextmanager
def zero_grad_and_step(opt):
    opt.zero_grad(set_to_none=True)
    yield
    opt.step()


def compute_nrow(images):
    b, _, h, w = images.size()
    nrow = math.ceil(math.sqrt(h * b / w))
    return nrow


def mix_up(images_0, labels_0, alpha):
    b = images_0.size(0)
    perm = torch.randperm(b)
    images_1, labels_1 = images_0[perm], labels_0[perm]

    lam = torch.distributions.Beta(alpha, alpha).sample((b,)).to(images_0.device)
    lam = torch.max(lam, 1 - lam)

    images = weighted_sum(images_0, images_1, lam.view(b, 1, 1, 1))
    labels = weighted_sum(labels_0, labels_1, lam)

    return images, labels


def cut_mix(images_0, labels_0, alpha):
    b, _, h, w = images_0.size()
    perm = np.random.permutation(b)
    images_1, labels_1 = images_0[perm], labels_0[perm]

    lam = np.random.beta(alpha, alpha)
    lam = np.maximum(lam, 1 - lam)

    r_w = w * np.sqrt(1 - lam)
    r_h = h * np.sqrt(1 - lam)

    t = np.random.uniform(0, h - r_h)
    l = np.random.uniform(0, w - r_w)
    b = t + r_h
    r = l + r_w

    t, l, b, r = [np.round(p).astype(np.int32) for p in [t, l, b, r]]
    assert 0 <= t <= b <= h
    assert 0 <= l <= r <= w

    images_0[:, :, t:b, l:r] = images_1[:, :, t:b, l:r]
    images = images_0
    labels = weighted_sum(labels_0, labels_1, lam)

    return images, labels


def make_grid(images):
    return torchvision.utils.make_grid(images, nrow=compute_nrow(images))


def bin_loss(w, sigma=0.2):
    c = 0.5 / math.sqrt(2 * math.pi * sigma ** 2)

    left = torch.exp(-1 * (w + 1) ** 2 / (2 * sigma ** 2))
    right = torch.exp(-1 * (w - 1) ** 2 / (2 * sigma ** 2))

    loss = -torch.log(c * left + c * right + 1e-8)

    return loss


def plot_decision_boundary(x, y, predict, colors=COLORS):
    grid = np.stack(
        np.meshgrid(
            np.linspace(x[:, 0].min() - 1.0, x[:, 0].max() + 1.0),
            np.linspace(x[:, 1].min() - 1.0, x[:, 1].max() + 1.0),
        ),
        -1,
    )
    z = predict(grid.reshape((-1, 2))).reshape(grid.shape[:2])

    fig = plt.figure()
    plt.contourf(
        grid[:, :, 0],
        grid[:, :, 1],
        z,
        levels=np.arange(z.max() + 2) - 0.5,
        colors=colors,
        alpha=0.5,
    )
    plt.scatter(x[y == 0][:, 0], x[y == 0][:, 1], s=5, c=colors[0])
    plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1], s=5, c=colors[1])

    return fig


def one_hot(input, n, dtype=torch.float):
    return torch.eye(n, dtype=dtype, device=input.device)[input]


def clip_grad_norm(grads, max_norm, norm_type=2):
    grads = list(filter(lambda grad: grad is not None, grads))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    total_norm = torch.norm(
        torch.stack([torch.norm(grad.detach(), norm_type) for grad in grads]), norm_type
    )
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        grads = [grad * clip_coef for grad in grads]

    return grads


def clip_parameters(m: nn.Module, value: float):
    if value <= 0:
        raise ValueError(f"expected clip value {value} to be > 0")

    for p in m.parameters():
        p.data.clamp_(-value, value)


def cross_entropy(input, target, dim=-1, eps=1e-8):
    return -torch.sum(target * torch.log(input + eps), dim=dim)


def entropy(input, dim=-1, eps=1e-8):
    return cross_entropy(input, input, dim=dim, eps=eps)


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
