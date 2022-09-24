from __future__ import annotations

import multiprocessing as mp
from functools import partial
from typing import Callable, Generic, TypeVar

A = TypeVar('A')
B = TypeVar('B')

Cont = Callable[[A], None]


def debug(f, *args, **kwargs):
    try:
        f(*args, **kwargs)
    except Exception as e:
        print(e)


class Q(Generic[A]):
    init: Callable[[Cont[A]], None]

    def __init__(self, init) -> None:
        self.init = init

    def run(self, cont: Cont[A]):
        q = mp.Queue()
        # spawn(self.init, args=(,))

        p = mp.Process(target=partial(debug, f=self.init), args=(lambda x: q.put(x),))
        p.start()

        # self.init(cont)

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
    # ray.init()

    c = range_q(10).bind(filter(lambda x: x % 2 == 0)).map(lambda x: x * 10)
    c.run(print)


if __name__ == '__main__':
    main()
