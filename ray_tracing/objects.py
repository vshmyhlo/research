import torch

from ray_tracing.material import Material
from ray_tracing.ray import Ray
from ray_tracing.vector import normalize


class Object(object):
    def __init__(self, material: Material):
        self.material = material


class Intersection(object):
    def __init__(self, object, t):
        self.object = object
        self.t = t


class ObjectList(object):
    def __init__(self, objects):
        self.objects = objects

    def intersects(self, ray: Ray):
        intersection = None

        for object in self.objects:
            t = object.intersects(ray)
            if t is None:
                continue
            if t < 0.001:
                continue

            if intersection is None:
                intersection = Intersection(object, t)
            elif t < intersection.t:
                intersection = Intersection(object, t)

        return intersection


class Sphere(Object):
    def __init__(self, center, radius, material: Material):
        super().__init__(material=material)

        self.center = center
        self.radius = radius

    def intersects(self, ray: Ray):
        sr = ray.origin - self.center

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

    def normal_at(self, position):
        return normalize(position - self.center)
