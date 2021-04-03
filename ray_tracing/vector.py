import torch


def vector(x, y, z):
    return torch.tensor([x, y, z], dtype=torch.float)


def normalize(v):
    return v / v.norm()


def random_unit():
    return normalize(vector(0, 0, 0).uniform_(-1.0, 1.0))


def reflect(v, n):
    return v - 2 * n * v.dot(n)


def random_in_hemisphere(n):
    v = random_unit()

    if v.dot(n) > 0.0:
        return v
    else:
        return -v
