from select import select

r = open("./read.txt")

a, b, c = select([r], [], [], 10)
print(a, b, c)
