def gen():
    print("(gen) init")
    for i in range(2):
        print("(gen) before yield")
        yield i
        print("(gen) after yield")


print("before init")
g = gen()
print("after init")
print()

print("before next")
next(g)
print("before next")
next(g)
print("before next")
next(g)
