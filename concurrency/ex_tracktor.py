import trio
import functools
import tractor


async def embed(it):
    for x in it:
        await trio.sleep(1)
        yield x


async def main():
    data = list(range(10))
    async with tractor.open_nursery() as n:
        # embs = embed(data)

        portal = await n.start_actor(
            "donny",
            enable_modules=[__name__],
        )

        async with portal.open_stream_from(embed, it=data) as stream:
            async for letter in stream:
                print(letter)


if __name__ == "__main__":
    trio.run(main)
