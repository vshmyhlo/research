import { ap, Monad } from './monad'

type State<A> =
    | {
          type: 'Some'
          value: A
      }
    | {
          type: 'None'
      }

class Maybe<A> implements Monad<A> {
    state: State<A>

    static some<A>(a: A) {
        return new Maybe({ type: 'Some', value: a })
    }

    static none<A>() {
        return new Maybe<A>({ type: 'None' })
    }

    constructor(state: State<A>) {
        this.state = state
    }

    map<B>(f: (a: A) => B): Maybe<B> {
        switch (this.state.type) {
            case 'Some':
                return Maybe.some(f(this.state.value))
            case 'None':
                return Maybe.none()
        }
    }

    bind<B>(f: (a: A) => Maybe<B>): Maybe<B> {
        switch (this.state.type) {
            case 'Some':
                const m = f(this.state.value)
                switch (m.state.type) {
                    case 'Some':
                        return Maybe.some(m.state.value)
                    case 'None':
                        return Maybe.none()
                }
            case 'None':
                return Maybe.none()
        }
    }
}

function main() {
    const a = Maybe.some('hello')
    const b = Maybe.some('world')
    const c = a.bind((a) => b.map((b) => a + ' ' + b))
    const d = ap(
        Maybe.some((a: string) => a + '!'),
        c
    )

    console.log(d)
}

main()
