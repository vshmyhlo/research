import time
from socket import *


def main():
    sock = socket(AF_INET, SOCK_STREAM)
    sock.bind(("", 9999))
    sock.listen(1)

    client, addr = sock.accept()
    print("accept")
    time.sleep(1)
    client.send(b"sup")
    print("send")
    client.close()
    print("close")


if __name__ == "__main__":
    main()
