import torch
from dataclasses import dataclass


@dataclass
class Light(object):
    position: torch.Tensor
    color: torch.Tensor
