from contextlib import contextmanager, asynccontextmanager
import asyncio


@contextmanager
def context():
    try:
        yield
    except Exception as e:
        print(type(e))
    finally:
        print("context done")


@asynccontextmanager
async def acontext():
    try:
        yield
    except Exception as e:
        print(type(e))
    finally:
        print("acontext done")


def gen():
    with context():
        yield 1
        yield 2
        yield 3


async def agen():
    async with acontext():
        yield 1
        yield 2
        yield 3


async def main():
    g = gen()
    print(next(g))
    g.close()

    ag = agen()
    print(await anext(ag))
    await ag.aclose()

    print("main done")


if __name__ == "__main__":
    asyncio.run(main())
