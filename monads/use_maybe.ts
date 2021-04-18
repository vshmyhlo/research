import { Maybe, Some, None, ap } from './maybe'

const xs: Maybe<number>[] = [new Some(2), new Some(4), new None()]
function f(x: number) {
    return x * 10
}
function g(x: number) {
    return x < 30 ? new Some(x) : new None()
}

console.log(xs)
console.log(xs.map((m) => m.fmap(f).bind(g)))

const xy: Maybe<(x: number) => string>[] = [
    new Some((x: number) => (x * 10).toString()),
    new None(),
]

console.log(xy)
console.log(xy.map((m) => ap(m, new Some(10))))
