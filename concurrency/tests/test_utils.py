from utils import merge, merge2, to_list, IterQueue
import asyncio
import pytest


async def make_it(n):
    for i in range(n):
        yield i


# @pytest.mark.asyncio
# async def test_merge():
#     it = merge([make_it(1), make_it(2), make_it(3)])
#     assert sorted(await to_list(it)) == [0, 0, 0, 1, 1, 2]
#     assert sorted(await to_list(it)) == []


# @pytest.mark.asyncio
# async def test_merge2():
#     async with asyncio.TaskGroup() as tg:
#         it = merge2([make_it(1), make_it(2), make_it(3)], tg)
#         assert sorted(await to_list(it)) == [0, 0, 0, 1, 1, 2]
#         assert sorted(await to_list(it)) == []


@pytest.mark.asyncio
async def test_queue():
    async def feed():
        await queue.feed(make_it(10))
        await queue.close()

    async with asyncio.TaskGroup() as tg:
        queue = IterQueue()
        tg.create_task(feed())
        assert sorted(await to_list(queue)) == [0, 0, 0, 1, 1, 2]
        assert sorted(await to_list(queue)) == []
