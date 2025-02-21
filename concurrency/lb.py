import asyncio
import multiprocessing as mp
import numpy as np
import time
from pprint import pprint
from contextlib import asynccontextmanager
from typing import (
    Dict,
    Any,
    AsyncIterable,
    Awaitable,
    Optional,
    ParamSpec,
    Callable,
    Sequence,
    TypeVar,
    Generic,
    AsyncIterator,
    Tuple,
)


P = ParamSpec("P")
T = TypeVar("T")
U = TypeVar("U")


async def iter_queue(queue: asyncio.Queue[Optional[T]]) -> AsyncIterator[T]:
    while True:
        x = await queue.get()
        if x is None:
            break
        yield x


async def pool_imap(fs: Sequence[Callable[[T], Awaitable[U]]], inputs: AsyncIterable[T]) -> AsyncIterator[U]:
    async def feed_input():
        i = 0
        async for input in inputs:
            await input_queue.put((i, input))
            i += 1
        for _ in fs:
            await input_queue.put(None)

    async def loop_workers():
        await asyncio.gather(*[loop_worker(f) for f in fs])
        await output_queue.put(None)

    async def loop_worker(f: Callable[[T], Awaitable[U]]):
        async for i, input in iter_queue(input_queue):
            await output_queue.put((i, await f(input)))

    async with asyncio.TaskGroup() as tg:
        input_queue = asyncio.Queue[Optional[Tuple[int, T]]](len(fs) * 2)
        output_queue = asyncio.Queue[Optional[Tuple[int, U]]](len(fs) * 2)

        tg.create_task(feed_input())
        tg.create_task(loop_workers())

        ready: Dict[int, U] = {}
        next_i = 0
        async for i, output in iter_queue(output_queue):
            ready[i] = output
            while next_i in ready:
                yield ready.pop(next_i)
                next_i += 1


async def load_balance(fs: Sequence[Callable[P, Awaitable[T]]]) -> Callable[P, Awaitable[T]]:
    pool = ResourcePool(fs)

    async def call(*args: P.args, **kwargs: P.kwargs) -> T:
        async with pool.get() as f:
            return await f(*args, **kwargs)

    return call


class ResourcePool(Generic[T]):
    def __init__(self, resources: Sequence[T]):
        self._resources = asyncio.Queue[T](len(resources))
        for r in resources:
            self._resources.put_nowait(r)

    @asynccontextmanager
    async def get(self):
        resource = await self._resources.get()
        try:
            yield resource
        finally:
            await self._resources.put(resource)


class F:
    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.num_calls = 0
        self.total_time = 9

    async def __call__(self, i) -> Any:
        t = time.time()
        await asyncio.sleep(self.delay)
        self.num_calls += 1
        self.total_time += time.time() - t
        print(f"done {i}")
        return i

    @property
    def stats(self):
        return {
            "delay": self.delay,
            "num_calls": self.num_calls,
            "total_time": self.total_time,
        }


async def gen():
    for i in range(100):
        yield i


async def main():
    fs = [F(0.1), F(0.2), F(0.4)]

    iis = []
    async for i in pool_imap(fs, gen()):
        iis.append(i)

    print(iis)
    assert iis == list(range(100))

    for f in fs:
        pprint(f.stats)


if __name__ == "__main__":
    asyncio.run(main())
