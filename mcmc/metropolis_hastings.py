import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from tqdm import tqdm


N = 100000


def f(x):
    a = 0.5
    return a * norm.pdf(x, -1.5, 0.75) + (1 - a) * norm.pdf(x, 1.5, 1 / 0.75)


def main():
    p = 0
    s = []

    pbar = tqdm(total=N)
    while len(s) < N:
        n = np.random.normal(p, 1)
        a = f(n) / f(p)

        u = np.random.uniform()
        if u <= a:
            s.append(n)
            p = n
            pbar.update()

    x = np.linspace(-5, 5, 1000)
    plt.hist(s, density=True, bins=1000)
    plt.plot(x, f(x))
    plt.show()


if __name__ == "__main__":
    main()
