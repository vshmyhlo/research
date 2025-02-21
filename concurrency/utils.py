import asyncio
from contextlib import asynccontextmanager

from typing import Generator, Generic, Awaitable, Any, List, Optional, Tuple

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


async def merge(its: Sequence[AsyncIterator[T]]) -> AsyncIterator[T]:
    task_to_iter = {asyncio.ensure_future(anext(it)): it for it in its}

    while task_to_iter:
        done, _ = await asyncio.wait(
            task_to_iter.keys(),
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in done:
            it = task_to_iter.pop(task)

            try:
                value = await task
            except StopAsyncIteration:
                continue
            else:
                yield value
                task_to_iter[asyncio.ensure_future(anext(it))] = it


# async def buffer():
#     pass


def merge2(
    its: Sequence[AsyncIterator[T]],
    tg: asyncio.TaskGroup,
) -> AsyncIterable[T]:
    async def feed_queues():
        feeds = [queue.feed(it) for it in its]
        await asyncio.gather(*feeds)
        await queue.close()

    queue = IterQueue(1)
    tg.create_task(feed_queues())
    return queue


class QueueClosed(Exception):
    pass


class IterQueue(Generic[T]):
    def __init__(self, maxsize: int = 0) -> None:
        self._queue = asyncio.Queue[Optional[T]](maxsize=maxsize)

    async def feed(self, it: AsyncIterator[T]):
        async for v in it:
            assert v is not None
            await self._queue.put(v)

    async def close_consumer(self):
        await self._queue.put(None)

    async def __aiter__(self) -> AsyncIterator[T]:
        while True:
            v = await self._queue.get()
            if v is None:
                break
            yield v


async def to_list(it: AsyncIterable[T]) -> List[T]:
    res = []
    async for x in it:
        res.append(x)
    return res


# async def queue(maxsize: int = 0) -> Tuple[ProduceContext, ConsumeContext]:
#     @asynccontextmanager
#     async def produce():
#         try:
#             yield write
#         finally:
#             close()

#     @asynccontextmanager
#     async def consume():
#         yield read

#     yield produce(), consume()


class Send(Generic[T]):
    def __init__(self, queue: asyncio.Queue[T]) -> None:
        self._queue = queue
        self._closed = False

    def close(self):
        self._closed = True

    async def feed(self, it: AsyncIterable[T]):
        assert not self._closed

        async for value in it:
            assert value is not None
            await self._queue.put(value)


class Recv(Generic[T]):
    def __init__(self, queue: asyncio.Queue[T]) -> None:
        self._queue = queue
        self._closed = False

    def close(self):
        self._closed = True

    async def __aiter__(self) -> AsyncIterator[T]:
        assert not self._closed

        while True:
            value = await self._queue.get()
            if value is None:
                break
            yield value


def queue(maxsize: int = 0) -> Tuple[Send[T], Recv[T]]:
    queue = asyncio.Queue[T](maxsize=maxsize)
    send = Send(queue)
    recv = Recv(queue)

    return send, recv
