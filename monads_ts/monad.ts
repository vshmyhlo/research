

export interface Monad<A> {
    map<B>(f: (a: A) => B): Monad<B>
    bind<B>(f: (a: A) => Monad<B>): Monad<B>
}