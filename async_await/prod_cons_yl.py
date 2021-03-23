from collections import deque

from termcolor import colored

from async_await.scheduler_yl import Scheduler, switch


class AsyncQueueClosed(Exception):
    pass


class AsyncQueue:
    def __init__(self):
        self.items = deque()
        self.waiting = deque()
        self.closed = False

    async def put(self, item):
        if self.closed:
            raise AsyncQueueClosed()
        self.items.append(item)
        if self.waiting:
            sched.ready.append(self.waiting.popleft())

    async def get(self):
        while not self.items:
            if self.closed:
                raise AsyncQueueClosed()
            self.waiting.append(sched.current)
            sched.current = None
            await switch()
        return self.items.popleft()

    async def close(self):
        self.closed = True
        while self.waiting:
            sched.ready.append(self.waiting.popleft())


def print_green(*ms):
    ms = (colored(m, "green") for m in ms)
    print(*ms)


def print_red(*ms):
    ms = (colored(m, "red") for m in ms)
    print(*ms)


async def producer(q: AsyncQueue, count):
    for n in range(count):
        print_green("producing", n)
        await q.put(n)
        await sched.sleep(1)

    print_green("producer done")
    await q.close()


async def consumer(q: AsyncQueue):
    while True:
        try:
            n = await q.get()
        except AsyncQueueClosed:
            break
        print_red("consuming", n)

    print_red("consumer done")


async def put():
    print_green("before put")
    await q.put(1)
    await q.put(2)
    print_green("after put")
    print_green("before get")
    x = await q.get()
    print_green("after get ({})".format(x))
    print_green("before close".format(x))
    await q.close()
    print_green("after close".format(x))


async def get():
    print_red("before get")
    try:
        x = await q.get()
    except AsyncQueueClosed:
        print_red("after exception")
    else:
        print_red("after get ({})".format(x))


sched = Scheduler()

print("check")
q = AsyncQueue()
for _ in range(3):
    sched.new_task(get())
sched.new_task(put())
sched.run()

print("prod/cons")
q = AsyncQueue()
sched.new_task(consumer(q))
sched.new_task(producer(q, 10))
sched.run()
