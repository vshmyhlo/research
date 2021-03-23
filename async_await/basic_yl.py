from async_await.scheduler_yl import Scheduler


async def countdown(n):
    for i in range(n, 0, -1):
        print("down", i)
        await sched.sleep(4)


async def countup(n):
    for i in range(n):
        print("up", i)
        await sched.sleep(1)


sched = Scheduler()
sched.new_task(countdown(5))
sched.new_task(countup(20))
sched.run()
