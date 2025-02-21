import asyncio
from typing import Awaitable, Any


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


async def main():
    s = await select(
        a=asyncio.sleep(1),
        b=asyncio.sleep(0.1),
    )

    match s:
        case S(a=res):
            print(f"got a={res}")
        case S(b=res):
            print(f"got b={res}")


if __name__ == "__main__":
    asyncio.run(main())
