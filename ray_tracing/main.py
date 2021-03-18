import os
import random
from functools import partial
from multiprocessing import Pool

import click
import matplotlib.pyplot as plt
import torch
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

import utils
from ray_tracing.camera import Camera
from ray_tracing.material import Metal, Diffuse, Light
from ray_tracing.objects import Sphere, ObjectList
from ray_tracing.ray import Ray
from ray_tracing.vector import vector, normalize


def randomize_objects():
    floor = Sphere(vector(0, -102, 1), 100, Diffuse(vector(1 / 3, 1 / 3, 1 / 3)))
    objects = [floor]

    for _ in range(8):
        floor_to_center = vector(0, -2, 1) - floor.center
        floor_to_center[0] += random.uniform(-4, 4)
        floor_to_center[2] += random.uniform(0, 8)

        radius = random.uniform(0.25, 1.5)
        center = floor.center + normalize(floor_to_center) * floor.radius + radius

        mat = random.choice([Diffuse, Metal])

        object = Sphere(center, radius, mat(vector(0, 0, 0).uniform_(0, 1)))
        objects.append(object)

    objects[8].material = Light(vector(1, 1, 1))

    return objects


@click.command()
@click.option("--size", type=click.INT, required=True)
@click.option("--k", type=click.INT, required=True)
@click.option("--steps", type=click.INT, required=True)
@click.option("--output-path", type=click.Path(), default="./ray_tracing/output")
def main(size, k, steps, output_path):
    # utils.random_seed(2**7)
    utils.random_seed(2 ** 10)
    size = size, size

    camera = Camera(vector(0, 0, -1))
    objects = ObjectList(randomize_objects())

    with Pool(os.cpu_count()) as pool:
        image = pool.imap(
            partial(
                render_row,
                size=size,
                camera=camera,
                objects=objects,
                k=k,
                max_steps=steps,
            ),
            range(size[0]),
        )
        image = list(tqdm(image, total=size[0]))

    image = torch.stack(image, 1)
    image = image.clamp(0, 1)
    image = image.flip(1)
    image = to_pil_image(image)

    os.makedirs(output_path, exist_ok=True)
    image.save(os.path.join(output_path, "{}_{}_{}.png".format(size[0], k, steps)))
    plt.imshow(image)
    plt.show()


def render_row(i, size, camera: Camera, objects: ObjectList, k, max_steps):
    utils.random_seed(i)

    row = torch.zeros(3, size[1], dtype=torch.float)
    for j in range(size[1]):
        for _ in range(k):
            y = (i + random.random()) / size[0]
            x = (j + random.random()) / size[1]

            ray = camera.ray_to_position(x, y)
            row[:, j] += ray_trace(ray, objects, max_steps=max_steps)

        row[:, j] /= k

    return row


def ray_trace(ray: Ray, objects: ObjectList, max_steps):
    if max_steps == 0:
        return vector(0, 0, 0)

    intersection = objects.intersects(ray)
    if intersection is None:
        return vector(0.2, 0.2, 0.2)

    position = ray.position_at(intersection.t)
    normal = intersection.object.normal_at(position)

    emitted = intersection.object.material.emit()
    reflection = intersection.object.material.reflect(ray, intersection.t, normal)
    if reflection is None:
        return emitted

    return emitted + reflection.attenuation * ray_trace(
        reflection.ray, objects, max_steps - 1
    )


if __name__ == "__main__":
    main()
