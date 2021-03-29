from socket import *


def main():
    sock = socket(AF_INET, SOCK_STREAM)
    sock.connect(("", 9999))
    print("connect")
    data = sock.recv(1024)
    print(b"recv " + data)


if __name__ == "__main__":
    main()
