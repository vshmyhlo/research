xs = iter(range(10))


def main():
    n = 0
    def it(xs):
        for x in xs:
            yield x
            n += 1
        
    list(it(xs))

    print( n)

main()
