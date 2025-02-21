
import numpy as np
import asyncio
from contextlib import aclosing
import time

def gen():
    for i in range(5):
        time.sleep(1)
        print(f'yield {i=}')
        yield i



async def process(x):
    async def a(x):
        await asyncio.sleep(np.random.uniform(0, 1))

    await asyncio.gather(a(x), a(x), a(x))
    print(f'processed {x=}')

async def main():
    for x in gen():
        await process(x)





if __name__ == '__main__':
    asyncio.run(main())
