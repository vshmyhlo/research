from __future__ import annotations

from typing import Callable, Generic, TypeVar

import ray
from ray.util.queue import Queue

A = TypeVar('A')
B = TypeVar('B')

Cont = Callable[[A], None]


def debug(f, *args, **kwargs):
    try:
        f(*args, **kwargs)
    except Exception as e:
        print(e)


def iter_queue(q):
    while True:
        x = q.get()
        if x is None:
            break

        yield x


@ray.remote
def populate(init, q):
    print('pop')
    init(lambda a: q.put(a))
    q.put(None)


class Q(Generic[A]):
    init: Callable[[Cont[A]], None]

    def __init__(self, init) -> None:
        self.init = init

    def run(self, cont: Cont[A]):
        print('run')
        q = Queue()
        populate.remote(self.init, q)
        for a in iter_queue(q):
            cont(a)

    def map(self, f: Callable[[A], B]) -> Q[B]:

        def gen(cont: Cont[B]):
            self.run(lambda a: cont(f(a)))

        return Q(gen)

    def bind(self, f: Callable[[A], Q[B]]) -> Q[B]:

        def gen(cont: Cont[B]):
            self.run(lambda a: f(a).run(cont))

        return Q(gen)


def range_q(n: int) -> Q[int]:

    def gen(cont):
        for i in range(n):
            cont(i)

    return Q(gen)


def filter(f):

    def aux(x):

        def gen(cont):
            if f(x):
                cont(x)

        return Q(gen)

    return aux


def main():
    ray.init()

    c = range_q(10).bind(filter(lambda x: x % 2 == 0)).map(lambda x: x * 10)
    # c = range_q(10).bind(filter(lambda x))
    c.run(print)


if __name__ == '__main__':
    main()
