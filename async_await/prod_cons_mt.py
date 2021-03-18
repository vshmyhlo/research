import queue
import threading
import time


def producer(q, count):
    for n in range(count):
        print("producing", n)
        q.put(n)
        time.sleep(1)

    print("producer done")
    q.put(None)


def consumer(q):
    while True:
        n = q.get()
        if n is None:
            break
        print("consuming", n)

    print("consumer done")


q = queue.Queue()
threading.Thread(target=producer, args=(q, 10)).start()
threading.Thread(target=consumer, args=(q,)).start()
