import time
from functools import partial

from async_await.scheduler import Scheduler


def countdown(n):
    if n > 0:
        print("down", n)
        sched.call_later(4, partial(countdown, n - 1))


def countup(n, i=0):
    if i < n:
        print("up", i)
        sched.call_later(1, partial(countup, n, i + 1))


sched = Scheduler()
sched.call_soon(partial(countdown, 5))
sched.call_soon(partial(countup, 20))
sched.run()
