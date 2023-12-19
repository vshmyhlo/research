import { Monad } from './monad'

class Reader<A, X> implements Monad<A> {
    run: (x: X) => A
    constructor(run: (x: X) => A) {
        this.run = run
    }

    map<B>(f: (a: A) => B): Reader<B, X> {
        return new Reader((x) => f(this.run(x)))
    }

    bind<B>(f: (a: A) => Reader<B, X>): Reader<B, X> {
        return new Reader((x) => {
            const m = f(this.run(x))
            return m.run(x)
        })
    }
}

function main() {
    const a = new Reader<number, number>((x) => 3 + x)
    const b = new Reader<number, number>((x) => 10 + x)
    const c = a.bind((a) => b.map((b) => a + b))

    console.log(c.run(2))
}

main()
