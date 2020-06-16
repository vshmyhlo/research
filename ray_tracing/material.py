from ray_tracing.ray import Ray

from ray_tracing.vector import random_unit, reflect, random_in_hemisphere


class Reflection(object):
    def __init__(self, ray, attenuation):
        self.ray = ray
        self.attenuation = attenuation


class Metal(object):
    def __init__(self, color):
        self.color = color

    def reflect(self, ray, t, normal):
        reflected = reflect(ray.direction, normal)
        reflected = Ray(ray.position_at(t), reflected)

        if reflected.direction.dot(normal) <= 0:
            return None

        return Reflection(reflected, self.color)


class DiffuseSphere(object):
    def __init__(self, color):
        self.color = color

    def reflect(self, ray, t, normal):
        ray = Ray(ray.position_at(t), normal + random_unit())

        return Reflection(ray, self.color)


class DiffuseHemisphere(object):
    def __init__(self, color):
        self.color = color

    def reflect(self, ray, t, normal):
        ray = Ray(ray.position_at(t), random_in_hemisphere(normal))

        return Reflection(ray, self.color)


Diffuse = DiffuseHemisphere
