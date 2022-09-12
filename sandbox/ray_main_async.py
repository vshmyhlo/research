import asyncio
import time

import ray


@ray.remote
def worker(q):
    while True:
        x = ray.get(q.get.remote())
        time.sleep(2)
        print(f'done {x}')


async def read():
    i = 0
    while True:
        await asyncio.sleep(0.2)
        yield i
        i += 1


@ray.remote
class Queue:

    def __init__(self, maxsize=0) -> None:
        self.q = asyncio.Queue(maxsize=maxsize)

    async def put(self, x):
        await self.q.put(x)

    async def get(self):
        return await self.q.get()


async def main():
    ray.init()

    q = Queue.remote(8)

    tasks = [worker.remote(q) for _ in range(4)]

    async for x in read():
        print(f'got {x}')
        await q.put.remote(x)

    ray.get(tasks)


if __name__ == '__main__':
    asyncio.run(main())
