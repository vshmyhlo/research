from abc import ABC, abstractmethod

import torch
from dataclasses import dataclass

from ray_tracing.utils import Material, Ray, normalize


@dataclass
class Object(ABC):
    material: Material

    @abstractmethod
    def ray_intersection(self, ray: Ray):
        raise NotImplementedError

    @abstractmethod
    def normal(self, position):
        raise NotImplementedError


@dataclass
class Sphere(Object):
    center: torch.Tensor
    radius: float

    def ray_intersection(self, ray: Ray):
        sr = ray.orig - self.center

        a = torch.dot(ray.direction, ray.direction)
        b = 2 * torch.dot(ray.direction, sr)
        c = torch.dot(sr, sr) - self.radius**2

        disc = b**2 - 4 * a * c
        if disc < 0:
            return None

        t = (-b - torch.sqrt(disc)) / (2 * a)
        if t <= 0:
            return None

        return t

    def normal(self, position):
        return normalize(position - self.center)
