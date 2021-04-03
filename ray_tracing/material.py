from ray_tracing.ray import Ray
from ray_tracing.vector import random_in_hemisphere, random_unit, reflect, vector


class Material(object):
    def reflect(self, ray, t, normal):
        raise NotImplementedError

    def emit(self):
        raise NotImplementedError


class Metal(Material):
    def __init__(self, color):
        self.color = color

    def reflect(self, ray, t, normal):
        reflected = reflect(ray.direction, normal)
        reflected = Ray(ray.position_at(t), reflected)

        if reflected.direction.dot(normal) <= 0:
            return None

        return Reflection(reflected, self.color)

    def emit(self):
        return vector(0, 0, 0)


class Diffuse(Material):
    def __init__(self, color, mode="hemi"):
        self.color = color
        self.mode = mode

    def reflect(self, ray, t, normal):
        if self.mode == "sphere":
            return self.reflect_sphere(ray, t, normal)
        elif self.mode == "hemi":
            return self.reflect_hemi(ray, t, normal)
        else:
            raise ValueError("invalid mode {}".format(self.mode))

    def reflect_sphere(self, ray, t, normal):
        ray = Ray(ray.position_at(t), normal + random_unit())

        return Reflection(ray, self.color)

    def reflect_hemi(self, ray, t, normal):
        ray = Ray(ray.position_at(t), random_in_hemisphere(normal))

        return Reflection(ray, self.color)

    def emit(self):
        return vector(0, 0, 0)


class Light(Material):
    def __init__(self, color):
        self.color = color

    def reflect(self, ray, t, normal):
        return None

    def emit(self):
        return self.color


class Reflection(object):
    def __init__(self, ray, attenuation):
        self.ray = ray
        self.attenuation = attenuation
