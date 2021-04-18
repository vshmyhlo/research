type Stack<T> = T[]

class State<S, T> {
    constructor(public f: (s: S) => [T, S]) {}

    bind<U>(f: (x: T) => State<S, U>): State<S, U> {
        return new State((s1: S) => {
            const [v, s2] = this.f(s1)
            return f(v).f(s2)
        })
    }
}

const push = (x: number) =>
    new State<Stack<number>, null>((stack) => [null, [x, ...stack]])

const pop = new State(([head, ...tail]) => [head, tail])

const manip = pop
    .bind(() => pop)
    .bind((x) => push(x * 10))
    .bind(() => pop)

console.log(manip.f([0, 1, 2]))
