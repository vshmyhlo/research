import asyncio
import time
from typing import (
    AsyncIterable,
    Tuple,
    Sequence,
    TypeVar,
    Optional,
    AsyncIterator,
    Iterable,
)
from contextlib import asynccontextmanager
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Iterator
from typing import Awaitable, Generic
import aiomultiprocess
import ray


T = TypeVar("T")


@ray.remote
class SlowProcessors(Generic[T]):
    def __init__(self, t: float, fail: bool = False) -> None:
        self.t = t
        self.fail = fail

    def process(self, x: T) -> T:
        if self.fail and x == 8:
            fail
        time.sleep(self.t)
        return x


async def load_data():
    for x in range(10):
        # if x == 8:
        #     fail
        yield x


async def main():
    index = Index()
    data = load_data()
    await index.index_batch(data)


class Index(Generic[T]):
    def __init__(self) -> None:
        self.embedder = SlowProcessors.remote(0.1)
        self.vector_db = SlowProcessors.remote(1)
        self.emb_storage = SlowProcessors.remote(1)

    async def index_batch(self, data: AsyncIterable[T]):
        (embs1, embs2), feed_task = multiplex(self.embed(data), 2)
        await asyncio.gather(
            feed_task,
            asyncio.create_task(self.index_vector_db(embs1)),
            asyncio.create_task(self.index_emb_storage(embs2)),
        )

    async def embed(self, data: AsyncIterable[T]):
        async with dummy_context("embed done"):
            async for x in data:
                await self.embedder.process.remote(x)
                yield x

    async def index_emb_storage(self, data: AsyncIterable[T]):
        async with dummy_context("index_emb_storage done"):
            async for x in data:
                await self.emb_storage.process.remote(x)
                print(f"index_emb_storage {x}")

    async def index_vector_db(self, data: AsyncIterable[T]):
        async with dummy_context("index_vector_db done"):
            async for x in data:
                await self.vector_db.process.remote(x)
                print(f"index_vector_db {x}")


def multiplex(
    it: AsyncIterable[T],
    n: int,
) -> Tuple[Sequence[AsyncIterator[T]], Awaitable]:
    async def feed_queues():
        async for x in it:
            assert x is not None
            await put_queues(x)
            print(f"put {x}")
        await put_queues(None)

    async def put_queues(x: Optional[T]):
        tasks = [asyncio.create_task(q.put(x)) for q in qs]
        await asyncio.gather(*tasks)

    async def iter_queue(q: asyncio.Queue[Optional[T]]) -> AsyncIterator[T]:
        while True:
            v = await q.get()
            if v is None:
                break
            yield v

    qs: Sequence[asyncio.Queue[Optional[T]]] = [asyncio.Queue(5) for _ in range(n)]
    feed_task = asyncio.create_task(feed_queues())
    return [iter_queue(q) for q in qs], feed_task


@asynccontextmanager
async def dummy_context(msg):
    try:
        yield
    finally:
        print(msg)


if __name__ == "__main__":
    asyncio.run(main())
