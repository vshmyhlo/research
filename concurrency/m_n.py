import asyncio
from utils import merge, IterQueue
from typing import Generator, Generic, Awaitable, Any

import random
from typing import (
    AsyncIterator,
    TypeVar,
    Sequence,
    Iterator,
    Tuple,
    Literal,
    Union,
    AsyncIterable,
)
import time


T = TypeVar("T")


class S:
    def __init__(self, k: str, v: Any) -> None:
        self.__setattr__(k, v)


async def select(**kwargs: Awaitable) -> S:
    mapping = {asyncio.ensure_future(f): k for k, f in kwargs.items()}
    (done, *_), _ = await asyncio.wait(
        mapping.keys(),
        return_when=asyncio.FIRST_COMPLETED,
    )
    return S(mapping[done], await done)


async def producer():
    for x in range(10):
        yield x


async def consumer(name, data: AsyncIterable):
    async for x in data:
        await asyncio.sleep(1)
        yield (name, x)


async def naive():
    async def produce():
        await asyncio.gather(*[queue1.feed(producer()) for _ in range(m)])
        await asyncio.gather(*[queue1.close_consumer() for _ in range(n)])

    async def consume():
        await asyncio.gather(
            *[queue2.feed(consumer(f"cons{i}", queue1)) for i in range(n)]
        )
        await queue2.close_consumer()
        await queue2.close_consumer()

    m, n = 3, 5
    queue1 = IterQueue(10)
    queue2 = IterQueue(10)

    t = time.time()

    async with asyncio.TaskGroup() as tg:
        tg.create_task(produce())
        tg.create_task(consume())

        async for out in queue2:
            print(out)

    print(time.time() - t)


async def better():
    async def produce():
        await asyncio.gather(*[queue1.feed(producer()) for _ in range(m)])
        await asyncio.gather(*[queue1.close_consumer() for _ in range(n)])

    async def consume():
        await asyncio.gather(
            *[queue2.feed(consumer(f"cons{i}", queue1)) for i in range(n)]
        )
        await queue2.close_consumer()

    m, n = 3, 5
    queue1 = IterQueue(10)
    queue2 = IterQueue(10)

    t = time.time()

    async with asyncio.TaskGroup() as tg:
        for _ in range(m):
            tg.create_task(queue1.clone().feed(producer))

        tg.create_task(produce())
        tg.create_task(consume())

        async for out in queue2:
            print(out)

    print(time.time() - t)


async def main():
    await naive()
    # await better()


if __name__ == "__main__":
    asyncio.run(main())
