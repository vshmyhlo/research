import { State } from './state'

type Stack<T> = T[]

const push = (x: number) =>
    new State<Stack<number>, null>((stack) => [null, [x, ...stack]])

const pop = new State(([head, ...tail]) => [head, tail])

const manip = pop
    .bind(() => pop)
    .bind((x) => push(x * 10))
    .bind(() => pop)

console.log(manip.f([0, 1, 2]))
