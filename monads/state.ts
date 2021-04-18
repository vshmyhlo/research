export class State<S, T> {
    constructor(public f: (s: S) => [T, S]) {}

    bind<U>(f: (x: T) => State<S, U>): State<S, U> {
        return new State((s1: S) => {
            const [v, s2] = this.f(s1)
            return f(v).f(s2)
        })
    }
}
