export abstract class Maybe<T> {
    abstract fmap<U>(f: (x: T) => U): Maybe<U>
    abstract bind<U>(f: (x: T) => Maybe<U>): Maybe<U>
}

export class Some<T> extends Maybe<T> {
    constructor(private value: T) {
        super()
    }

    fmap<U>(f: (x: T) => U): Some<U> {
        return new Some(f(this.value))
    }

    bind<U>(f: (x: T) => Maybe<U>): Maybe<U> {
        return f(this.value)
    }
}

export class None<T> extends Maybe<T> {
    fmap<U>(f: (x: T) => U): None<U> {
        return new None()
    }

    bind<U>(f: (x: T) => Maybe<U>): Maybe<U> {
        return new None()
    }
}

export function ap<T extends (x: U) => V, U, V>(
    a: Maybe<T>,
    b: Maybe<U>
): Maybe<V> {
    return a.bind((f) => b.fmap((v) => f(v)))
}
