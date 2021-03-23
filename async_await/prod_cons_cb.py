import queue
import threading
from collections import deque
from functools import partial

from async_await.scheduler_cb import Scheduler


class AsyncQueue:
    def __init__(self):
        self.items = deque()
        self.waiting = deque()

    def put(self, item):
        self.items.append(item)
        if self.waiting:
            func = self.waiting.popleft()
            sched.call_soon(func)

    def get(self, callback):
        if self.items:
            callback(self.items.popleft())
        else:
            self.waiting.append(partial(self.get, callback))


def producer(q: AsyncQueue, count):
    def _produce(n):
        if n < count:
            print("producing", n)
            q.put(n)
            sched.call_later(1, partial(_produce, n + 1))
        else:
            print("producer done")
            q.put(None)

    _produce(0)


def consumer(q: AsyncQueue):
    def _consume(n):
        if n is None:
            print("consumer done")
        else:
            print("consuming", n)
            q.get(_consume)

    q.get(_consume)


sched = Scheduler()
q = AsyncQueue()
sched.call_soon(partial(producer, q, 10))
sched.call_soon(partial(consumer, q))
sched.run()
