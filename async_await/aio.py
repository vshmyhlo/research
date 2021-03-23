import asyncio

asyncio.sleep()


class Awaitable:
    def __await__(self):
        yield


async def gen():
    print("(gen) init")
    for i in range(2):
        print("(gen) before yield")
        await Awaitable()
        print("(gen) after yield")


print("before init")
g = gen()
print("after init")
print()

print("before next")
g.send(None)
print("before next")
g.send(None)
print("before next")
g.send(None)
