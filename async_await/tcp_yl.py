import socket

from async_await.scheduler_yl import Scheduler


async def tcp_server(addr):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(addr)
    sock.listen(1)
    while True:
        client, addr = await sched.accept(sock)
        sched.new_task(echo_handler(client))


async def echo_handler(sock):
    while True:
        data = await sched.recv(sock, 1024)
        if not data:
            break
        await sched.sleep(0.25)
        await sched.send(sock, b"got: " + data)
    print("connection closed")


sched = Scheduler()
sched.new_task(tcp_server(("", 9999)))
sched.run()
