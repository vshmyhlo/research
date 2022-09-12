import asyncio


async def read():
    i = 0
    while True:
        await asyncio.sleep(0.2)
        yield i
        i += 1


async def worker(x):
    await asyncio.sleep(2)
    print(f'work done {x}')


async def main():
    input = read()
    in_progress = set()
    async for x in input:

        while len(in_progress) > 3:
            _, in_progress = await asyncio.wait(in_progress, return_when=asyncio.FIRST_COMPLETED)

        in_progress.add(asyncio.create_task(worker(x)))
        print(f"submit {x}")


if __name__ == '__main__':
    asyncio.run(main())
