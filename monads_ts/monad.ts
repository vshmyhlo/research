export interface Monad<A> {
    map<B>(f: (a: A) => B): Monad<B>
    bind<B>(f: (a: A) => Monad<B>): Monad<B>
}

export function ap<A, B>(m1: Monad<(a: A) => B>, m2: Monad<A>): Monad<B> {
    return m1.bind((f) => m2.map(f))
}
