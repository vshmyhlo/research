import asyncio
import os
from concurrent.futures import ProcessPoolExecutor
import time
from typing import Iterator
from typing import Awaitable
import aiomultiprocess
import ray


def cpu_task():
    print(f"cpu_task {os.getpid()}")
    time.sleep(5)
    print("cpu_task done")


async def async_cpu_task():
    print(f"cpu_task {os.getpid()}")
    time.sleep(5)
    print("cpu_task done")


class Worker:
    async def __aiter__(self):
        yield 1


# def spawn(f):
#     pass


class Spawn:
    def f():
        pass


class A:
    def __await__(self):
        yield
        return "A return value"


async def async_iter():
    for i in range(3):
        await asyncio.sleep(1)
        yield i


async def ticker():
    for i in range(10):
        await asyncio.sleep(1)
        print(f"ticker {i}")


async def mp_example():
    def wait_done():
        time.sleep(3)
        print("thread done")
        # f.set_result(1)

    # f = asyncio.Future()

    return await asyncio.to_thread(wait_done)

    # return f


async def main():
    print(f"main {os.getpid()}")

    # ex = ProcessPoolExecutor(os.cpu_count())

    ticker_t = asyncio.create_task(ticker())
    # t2 = asyncio.create_task(ticker())
    # t3 = asyncio.get_event_loop().run_in_executor(ex, cpu_task)
    # # t3 = await aiomultiprocess.Worker(async_cpu_task)
    # # t3 = mp_example()

    # await asyncio.gather(
    #     t1,
    #     t2,
    #     t3,
    # )

    # print("start")

    # print("a", await A())

    # it = async_iter()

    # async for x in Worker():
    #     print(x)

    # f = asyncio.Future()

    # data_1, data_2 = multiplex(data, 2)
    # del data

    # ticker_y = asyncio.create_task(ticker())

    # data = range(10)
    docs = range(10)
    # data = asyncio.create_task(embed(docs))

    embs = embed(docs)
    # embs_1, embs_2, feed_t = multiplex(embs)
    del embs

    vector_db_t = asyncio.create_task(insert_vector_db(embs_1))
    emb_storage_t = asyncio.create_task(insert_emb_storage(embs_2))

    await asyncio.gather(
        vector_db_t,
        emb_storage_t,
        feed_t,
    )

    # await t1
    # await t2

    # for x in embeds:
    #     async

    # embs = []

    # tasks = []
    # for emb in embs:
    #     tasks.append(
    #         asyncio.gather(
    #             insert_vector_db(emb),
    #             insert_emb_storage(emb),
    #         )
    #     )


def multiplex(it):
    async def feed():
        for x in it:
            await a.put(x)
            await b.put(x)

    a = asyncio.Queue(1)
    b = asyncio.Queue(1)

    feed_t = asyncio.create_task(feed())

    return a, b, feed_t


# async def multiplex(it, n):
#     copies = mp

#     # def feed():
#     #     pass

#     t = asyncio.create_task(feed)

#     # async for x in it:
#     #     yield x
#     #     yield x

#     return copies


async def consume(it, name):
    async for x in it:
        print(f"got {x} in {name}")


async def cancel_future(f: asyncio.Future):
    await asyncio.sleep(3)
    f.cancel()


async def embed(docs):
    for d in docs:
        await asyncio.sleep(1)
        yield d


async def insert_vector_db(data: Iterator):
    a = VectorDB.remote()
    async for x in data:
        await a.process.remote(x)


async def insert_emb_storage(data: Iterator):
    a = EmbStorage.remote()
    async for x in data:
        await a.process.remote(x)


@ray.remote
class EmbStorage:
    def process(self, x):
        time.sleep(1)
        print(f"EmbStorage.process {x}")


@ray.remote
class VectorDB:
    def process(self, x):
        time.sleep(1)
        print(f"VectorDB.process {x}")


if __name__ == "__main__":
    asyncio.run(main())
