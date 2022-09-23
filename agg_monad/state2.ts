import { Stack } from 'immutable'

class State<S, A> {
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

// const join =
//     <S, A>(m: State<S, State<S, A>>): State<S, A> =>
//     (s: S) => {
//         const [nested, s2] = m(s)
//         return nested(s2)
//     }

// const put =
//     <S>(s: S) =>
//     (_: any) =>
//         [null, s] as [null, S]

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
