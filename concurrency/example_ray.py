import asyncio
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
import time
from typing import Iterator
from typing import Awaitable
import aiomultiprocess
import ray


T = TypeVar("T")


async def load_data():
    for x in range(10):
        yield x


async def main():
    index = Index()
    data = load_data()
    await index.index_batch(data)


class Index:
    def __init__(self) -> None:
        pass

    async def index_batch(self, data: AsyncIterable[T]):
        async with multiplex(self.embed(data), 2) as (embs1, embs2):
            await asyncio.gather(
                asyncio.create_task(self.index_vector_db(embs1)),
                asyncio.create_task(self.index_emb_storage(embs2)),
            )

    async def embed(self, data: AsyncIterable[T]):
        async for x in data:
            await asyncio.sleep(1)
            yield x

    async def index_emb_storage(self, data: AsyncIterable[T]):
        async for x in data:
            await asyncio.sleep(1)
            print(f"index_emb_storage {x}")

    async def index_vector_db(self, data: AsyncIterable[T]):
        async for x in data:
            await asyncio.sleep(1)
            print(f"index_vector_db {x}")


@asynccontextmanager
async def multiplex(
    it: AsyncIterable[T], n: int
) -> AsyncIterator[Tuple[AsyncIterable[T], ...]]:
    async def feed_queues():
        async for x in it:
            assert x is not None
            await put_queues(x)
        await put_queues(None)

    async def put_queues(x: Optional[T]):
        tasks = [asyncio.create_task(q.put(x)) for q in qs]
        await asyncio.gather(*tasks)

    async def iter_queue(q: asyncio.Queue[Optional[T]]) -> AsyncIterable[T]:
        while True:
            v = await q.get()
            if v is None:
                break
            yield v

    qs: Sequence[asyncio.Queue[Optional[T]]] = [asyncio.Queue(1) for _ in range(n)]
    feed_t = asyncio.create_task(feed_queues())
    yield tuple(iter_queue(q) for q in qs)
    await feed_t


if __name__ == "__main__":
    asyncio.run(main())
