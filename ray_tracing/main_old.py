from itertools import product

import matplotlib.pyplot as plt
import torch
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from ray_tracing.light import Light
from ray_tracing.material import Metal
from ray_tracing.objects import Sphere, Object
from ray_tracing.ray import Ray
from ray_tracing.scene import Scene
from ray_tracing.vector import vector, normalize


def build_view(size):
    def build_axis(size):
        step = 2 / size
        axis = torch.arange(-1, 1, step) + step / 2

        return axis

    y = build_axis(size[0])
    x = build_axis(size[1])
    yx = torch.stack(torch.meshgrid(y, x), 0)
    yx = torch.cat([yx, torch.zeros_like(yx[:1])])

    return yx


def main():
    size = 256, 256

    camera = vector(0, 0, -1)
    objects = [
        Sphere(vector(0, 0, 0), 0.5, Metal(vector(1, 0, 0))),
        Sphere(vector(0, -1, 1), 0.5, Metal(vector(0, 1, 0))),
    ]
    lights = [
        Light(vector(1.5, -0.5, -10), vector(1, 1, 1)),
    ]
    scene = Scene(
        camera=camera,
        objects=objects,
        lights=lights)

    view = build_view(size)
    image = torch.zeros(3, *size, dtype=torch.float)

    for i, j in tqdm(product(range(size[0]), range(size[1])), total=size[0] * size[1]):
        ray = Ray(camera, view[:, i, j] - camera)
        image[:, i, j] = ray_trace(ray, scene)

    image = to_pil_image(image)
    image.save('./ray_tracing/output.png')
    plt.imshow(image)
    plt.show()


def ray_trace(ray: Ray, scene: Scene):
    color = vector()

    ot = None
    for object in scene.objects:
        t = object.intersects(ray)
        if t is None:
            continue
        if ot is None:
            ot = object, t
        if t < ot[1]:
            ot = object, t

    if ot is None:
        return color

    object, t = ot
    del ot

    position = ray.position_at(t)
    normal = object.normal_at(position)
    color += color_at(object, position, normal, scene)

    return color


def color_at(object: Object, position, normal, scene: Scene):
    color = vector()
    color = object.material.ambient * color

    to_cam = scene.camera - position
    specular_k = 50

    for light in scene.lights:
        to_light = Ray(position, light.position - position)
        color += object.material.color * \
                 object.material.diffuse * \
                 max(torch.dot(normal, to_light.direction), 0)

        half_vector = normalize(to_light.direction + to_cam)
        color += light.color * \
                 object.material.specular * \
                 max(torch.dot(normal, half_vector), 0)**specular_k

    color = color.clamp(0, 1)

    return color


if __name__ == '__main__':
    main()
