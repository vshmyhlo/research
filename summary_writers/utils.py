import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image


def convert_scalar(value):
    if isinstance(value, (torch.Tensor, np.ndarray)):
        return value.item()

    return value


def convert_image(value):
    if isinstance(value, (torch.Tensor,)):
        return to_pil_image(value)

    return value


def sanitize_tag(tag):
    return tag.replace("/", "_")
