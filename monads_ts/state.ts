import { Stack } from 'immutable'

type State<S, V> = (s: S) => [V, S]

const map =
    <S, A, B>(m: State<S, A>, f: (a: A) => B): State<S, B> =>
    (s: S) => {
        const [a, s2] = m(s)
        return [f(a), s2]
    }

const bind =
    <S, A, B>(m: State<S, A>, f: (a: A) => State<S, B>) =>
    (s: S) => {
        const [a, s2] = m(s)
        return f(a)(s2)
    }

const then =
    <S, A, B>(m: State<S, A>, m2: State<S, B>) =>
    (s: S) => {
        const [_, s2] = m(s)
        return m2(s2)
    }

const join =
    <S, A>(m: State<S, State<S, A>>): State<S, A> =>
    (s: S) => {
        const [nested, s2] = m(s)
        return nested(s2)
    }

const put =
    <S>(s: S) =>
    (_: any) =>
        [null, s] as [null, S]

const push =
    <A>(a: A) =>
    (s: Stack<A>) =>
        [null as null, s.push(a)] as [null, Stack<A>]

const pop = <A>(s: Stack<A>) => {
    const a = s.peek()
    return [a, s.pop()] as [A, Stack<A>]
}

function main() {
    let state: State<Stack<number>, number> = push(0)
    state = then(state, push(1))
    state = then(state, push(2))
    state = then(state, push(3))
    state = then(
        state,
        bind(pop, (a) => bind(pop, (b) => push(a + b)))
    )
    state = then(state, pop)

    console.log(state(Stack()))
}

main()
