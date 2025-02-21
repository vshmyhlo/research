import time

import numpy as np
import ray


def gen():
    while True:
        time.sleep(np.random.uniform(0, 1))
        yield np.zeros((np.random.randint(1000), np.random.randint(1000)))


@ray.remote
class A:
    def __init__(self) -> None:
        self.g = gen()

    def get(self):
        x = next(self.g)
        t = time.time()

        return x, t


def main():
    ray.init()

    a = A.remote()

    for _ in range(10_000):
        pass

    while True:
        pass


if __name__ == "__main__":
    main()
