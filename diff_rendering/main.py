import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import click
from dataclasses import dataclass


@dataclass()
class Ray(object):
    orig: torch.Tensor
    dir: torch.Tensor


@dataclass()
class Sphere(object):
    cent: torch.Tensor
    rad: torch.Tensor
    col: torch.Tensor

    def intersects(self, ray: Ray):
        ray = Ray(ray.orig.unsqueeze(1), ray.dir.unsqueeze(1))
        self = Sphere(
            self.cent.unsqueeze(0), self.rad.unsqueeze(0), self.col.unsqueeze(0)
        )

        sr = ray.orig - self.cent

        a = dot(ray.dir, ray.dir)
        b = 2 * dot(ray.dir, sr)
        c = dot(sr, sr) - self.rad ** 2

        disc = b ** 2 - 4 * a * c

        t = torch.where(
            disc < 0,
            torch.full_like(disc, float("inf")),
            (-b - torch.sqrt(disc)) / (2 * a),
        )

        t = torch.where(
            t <= 0,
            torch.full_like(t, float("inf")),
            t,
        )

        t, i = t.min(1)

        i = torch.where(
            t == float("inf"),
            torch.full_like(i, -1),
            i,
        )

        return t, i


def dot(a, b, dim=-1):
    return torch.sum(a * b, dim=dim)


def build_pixel_grid(size):
    y = (torch.arange(0, 1, 1 / size) + 1 / size / 2) * 2 - 1
    x = (torch.arange(0, 1, 1 / size) + 1 / size / 2) * 2 - 1

    grid = torch.stack(torch.meshgrid(y, x), -1)
    grid = F.pad(grid, (0, 1))

    return grid


@click.command()
@click.option("--size", type=click.INT, required=True)
def main(size):
    camera = torch.tensor([0, 0, -1], dtype=torch.float)

    pixel_grid = build_pixel_grid(size)
    pixel_grid = pixel_grid.view(size * size, 3)

    center = torch.tensor([[0, -1.2, 2], [0, 1.2, 2]], dtype=torch.float)
    radius = torch.tensor([1, 1], dtype=torch.float)
    color = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float)
    object = Sphere(center, radius, color)

    ray = Ray(camera.unsqueeze(0), pixel_grid - camera.unsqueeze(0))

    image = trace(ray, object, max_steps=32)
    image = image.view(size, size, 3)

    image = TF.to_pil_image(image.permute(2, 0, 1))
    plt.imshow(image)
    plt.show()


def trace(ray, object, max_steps):
    t, i = object.intersects(ray)

    color = torch.where(
        (i < 0).unsqueeze(1),
        torch.zeros(i.size(0), 3),
        object.col[i],
    )

    return color


if __name__ == "__main__":
    main()
