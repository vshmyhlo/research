from contextlib import contextmanager
import asyncio


@contextmanager
def some_context():
    print("entering context")
    try:
        yield
    except Exception as e:
        print(f"exception {e}")
    finally:
        print("leaving context")


async def async_gen():
    with some_context():
        for _ in range(10):
            await asyncio.sleep(1)


def gen():
    with some_context():
        yield from range(10)


async def async_f():
    g = asyncio.create_task(async_gen())
    await asyncio.sleep(3)
    g.cancel()


def f():
    g = gen()
    next(g)
    next(g)
    next(g)


async def q_block():
    async def block():
        with some_context():
            q = asyncio.Queue()
            await q.get()

    t = asyncio.create_task(block())
    await asyncio.sleep(3)
    t.cancel()


asyncio.run(q_block())
