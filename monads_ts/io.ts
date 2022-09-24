import { Monad } from './monad'
class IO<A> implements Monad<A> {
    run: () => A

    constructor(f: () => A) {
        this.run = f
    }

    map<B>(f: (a: A) => B): IO<B> {
        return new IO(() => f(this.run()))
    }

    bind<B>(f: (a: A) => IO<B>): IO<B> {
        return new IO(() => f(this.run()).run())
    }
}

function main() {
    const a = new IO(() => {
        console.log('hello effect')
        return 'hello'
    })
    const b = new IO(() => {
        console.log('world effect')
        return 'world'
    })


    const c = a.bind((a) => b.map((b) => a + ' ' + b))

    console.log('run')
    console.log('result', c.run())
}

main()
