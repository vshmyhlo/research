import time

import ray


@ray.remote()
def gen():
    for i in range(10):
        print(f'yield {i}')
        yield i


def main():
    ray.init()

    g = gen.remote()
    print(g)

    time.sleep(3)


if __name__ == '__main__':
    main()
