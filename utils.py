import math

import torchvision


def compute_nrow(images):
    b, _, h, w = images.size()
    nrow = math.ceil(math.sqrt(h * b / w))

    return nrow


def make_grid(images):
    return torchvision.utils.make_grid(images, nrow=compute_nrow(images))
