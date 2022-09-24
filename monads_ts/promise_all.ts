function all<T>(ps: Promise<T>[]): Promise<T[]> {
    if (ps.length === 0) {
        return Promise.resolve([])
    }

    return ps[0].then((x) => all(ps.slice(1)).then((xs) => [x, ...xs]))
}

function delay() {
    return new Promise((res) => {
        setTimeout(res, Math.random() * 1000)
    })
}

function* range(n: number) {
    for (let i = 0; i < n; i++) {
        yield i
    }
}

function main() {
    const ps = Array.from(range(100)).map((x) => delay().then(() => x))

    console.time('total')
    all(ps).then((xs) => {
        console.log(xs)
        console.timeEnd('total')
    })
}

main()
