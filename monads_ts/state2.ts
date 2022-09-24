import { Stack } from 'immutable'
import { Monad } from './monad'

class State<S, A> implements Monad<A> {
    run: (s: S) => [A, S]

    constructor(run: (s: S) => [A, S]) {
        this.run = run
    }

    map<B>(f: (a: A) => B) {
        return new State((s: S) => {
            const [a, s2] = this.run(s)
            return [f(a), s2]
        })
    }

    bind<B>(f: (a: A) => State<S, B>) {
        return new State((s: S) => {
            const [a, s2] = this.run(s)
            return f(a).run(s2)
        })
    }

    then<B>(m: State<S, B>) {
        return new State((s: S) => {
            const [_, s2] = this.run(s)
            return m.run(s2)
        })
    }
}

const push = <A>(a: A) => new State((s: Stack<A>) => [null, s.push(a)])

const pop = new State<Stack<number>, number>(<A>(s: Stack<A>) => {
    const a = s.peek()
    return [a, s.pop()]
})

function main() {
    let state: State<Stack<number>, number> = push(0)
        .then(push(1))
        .then(push(2))
        .then(push(3))
        .then(pop.bind((a) => pop.bind((b) => push(a + b))))
        .then(pop)

    console.log(state.run(Stack()))
}

main()
