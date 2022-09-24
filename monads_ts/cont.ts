import { Monad } from './monad'
class Cont<A> implements Monad<A> {
    init: (cont: (a: A) => void) => void

    constructor(init: (cont: (a: A) => void) => void) {
        this.init = init
    }

    run(cont: (a: A) => void) {
        this.init(cont)
    }

    map<B>(f: (a: A) => B): Cont<B> {
        return new Cont<B>((cont) => this.run((a) => cont(f(a))))
    }

    bind<B>(f: (a: A) => Cont<B>): Cont<B> {
        return new Cont<B>((cont) => this.run((a) => f(a).run(cont)))
    }
}

function delay(): Cont<void> {
    return new Cont((cont) => setTimeout(cont, Math.random() * 1000))
}

function dup<A>(a: A): Cont<A> {
    return new Cont((cont) => {
        cont(a)
        cont(a)
    })
}

function range(n: number): Cont<number> {
    return new Cont((cont) => {
        for (let i = 0; i < n; i++) {
            cont(i)
        }
    })
}

// function main() {
//     const a = delay()
//         .map(() => 'hello')
//         .bind(dup)
//     const b = delay().map(() => 'world')
//     const c = a.bind((a) => b.map((b) => a + ' ' + b))
//     c.run((c) => console.log(c))
// }

function filter<A>(f: (a: A) => boolean): (a: A) => Cont<A> {
    return (a) =>
        new Cont((cont) => {
            if (f(a)) {
                cont(a)
            }
        })
}

function main() {
    const a = range(10)
        .bind(filter((a) => a % 2 == 0))
        .bind(dup)
    a.run((a) => console.log(a))
}

main()
