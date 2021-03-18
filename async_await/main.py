import time


def countdown(n):
    for i in range(n, 0, -1):
        print("down", i)
        time.sleep(1)


def countup(n):
    for i in range(n):
        print("up", i)
        time.sleep(1)


countdown(5)
countup(5)
