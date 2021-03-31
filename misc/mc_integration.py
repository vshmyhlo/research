import numpy as np
import scipy.stats

min, max = 0, 10

x = np.random.uniform(min, max, size=100000)
x = scipy.stats.norm(0, 1).pdf(x)


mask = (min <= x) & (x <= max)

x = x[mask]
i = x.sum() * (max - min) / x.shape[0]

print(i)


def f(x):
    return np.linalg.norm(x, ord=2, axis=-1) <= 1


x = np.random.uniform(-1, 1, size=(100000, 2))
v = 2 * 2
i = v * (1 / x.shape[0]) * f(x).sum()
print(i, np.pi * 1 ** 2)
