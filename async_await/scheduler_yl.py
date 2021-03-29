import heapq
import time
from collections import deque
from select import select


class Scheduler:
    def __init__(self):
        self.current = None
        self.ready = deque()
        self.sleeping = []
        self.read_waiting = {}
        self.write_waiting = {}
        self.sequence = 0

    def new_task(self, coro):
        self.ready.append(coro)

    async def sleep(self, delay):
        deadline = time.time() + delay
        self.sequence += 1
        heapq.heappush(self.sleeping, (deadline, self.sequence, self.current))
        self.current = None
        await switch()

    # async def read_wait(self, fileno):
    #     self.read_waiting[fileno] = self.current
    #     self.current = None
    #     await switch()
    #
    # async def write_wait(self, fileno):
    #     self.read_waiting[fileno] = self.current
    #     self.current = None
    #     await switch()

    async def recv(self, sock, maxbytes):
        self.read_waiting[sock] = self.current
        self.current = None
        await switch()
        return sock.recv(maxbytes)

    async def send(self, sock, data):
        self.write_waiting[sock] = self.current
        self.current = None
        await switch()
        return sock.send(data)

    async def accept(self, sock):
        self.read_waiting[sock] = self.current
        self.current = None
        await switch()
        return sock.accept()

    def run(self):
        while self.ready or self.sleeping or self.read_waiting or self.write_waiting:
            if not self.ready:
                if self.sleeping:
                    deadline, _, coro = self.sleeping[0]
                    timeout = deadline - time.time()
                    if timeout < 0:
                        timeout = 0
                else:
                    timeout = None

                can_read, can_write, _ = select(self.read_waiting, self.write_waiting, [], timeout)

                for fileno in can_read:
                    self.ready.append(self.read_waiting.pop(fileno))
                for fileno in can_write:
                    self.ready.append(self.write_waiting.pop(fileno))

                now = time.time()
                while self.sleeping:
                    deadline, _, coro = self.sleeping[0]
                    if now > deadline:
                        heapq.heappop(self.sleeping)
                        self.ready.append(coro)
                    else:
                        break

                # self.ready.append(coro)

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
