import torch
from dataclasses import dataclass


@dataclass
class Material(object):
    color: torch.Tensor
    ambient: float = 0.05
    diffuse: float = 1.
    specular: float = 1.

    # def color_at(self, position):
    #     self.color


@dataclass
class Ray(object):
    orig: torch.Tensor
    direction: torch.Tensor

    def position(self, t: float):
        return self.orig + t * self.direction


def vector(x=0, y=0, z=0):
    return torch.tensor([x, y, z], dtype=torch.float)


def normalize(vector):
    return vector / vector.norm()
