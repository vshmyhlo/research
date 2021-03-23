import heapq
import time
from collections import deque


class Scheduler:
    def __init__(self):
        self.current = None
        self.ready = deque()
        self.sleeping = []
        self.sequence = 0

    def new_task(self, coro):
        self.ready.append(coro)

    async def sleep(self, delay):
        deadline = time.time() + delay
        self.sequence += 1
        heapq.heappush(self.sleeping, (deadline, self.sequence, self.current))
        self.current = None
        await switch()

    def run(self):
        while self.ready or self.sleeping:
            if not self.ready:
                deadline, _, coro = heapq.heappop(self.sleeping)
                delta = deadline - time.time()
                if delta > 0:
                    time.sleep(delta)
                self.ready.append(coro)

            while self.ready:
                self.current = self.ready.popleft()
                try:
                    self.current.send(None)
                    if self.current:
                        self.ready.append(self.current)
                except StopIteration:
                    pass


class Awaitable:
    def __await__(self):
        yield


def switch():
    return Awaitable()
