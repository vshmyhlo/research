import Queue from './queue'

function buildPromise<T>(): [Promise<T>, (x: T) => void] {
    let resolve
    const p = new Promise<T>((res) => {
        resolve = res
    })
    return [p, resolve]
}

type Subscriber<T> = (x: T) => Promise<void>

class Stream<T> {
    subscribers: Subscriber<T>[]

    constructor() {
        this.subscribers = []
    }

    async emit(x: T) {
        for (const f of this.subscribers) {
            await f(x)
        }
    }

    subscribe(f: Subscriber<T>) {
        this.subscribers.push(f)
    }

    map<U>(f: (x: T) => Promise<U>) {
        const s = new Stream()
        this.subscribe(async (x) => s.emit(await f(x)))
        return s
    }
}

async function task(x: number) {
    await sleep(1000)
    return x * 10
}

async function sleep(delay: number) {
    return new Promise((res) => setTimeout(res, delay))
}

async function worker<T, U>(
    task: (x: T) => Promise<U>,
    q: Queue<{ arg: T; res: (x: U) => void }>
) {
    while (true) {
        const { arg, res } = await q.get()
        res(await task(arg))
    }
}

function parallel<T, U>(
    task: (x: T) => Promise<U>,
    n: number
): (x: T) => Promise<U> {
    const q = new Queue<{ arg: T; res: (x: U) => void }>()

    for (let i = 0; i < n; i++) {
        worker(task, q)
    }

    return async (x) => {
        const [p, res] = buildPromise<U>()
        await q.put({
            arg: x,
            res,
        })
        return await p
    }
}

async function main() {
    const a = new Stream<number>()

    const b = a.map(task)
    // const b = a.map(parallel(task, 4))
    b.subscribe(async (x) => console.log('got value', x))

    const tasks = []
    for (let i = 0; i < 10; i++) {
        tasks.push(sleep(200).then(() => a.emit(i)))
    }
    for (const t of tasks) {
        await t
    }
}

// async function main() {
//     const f = parallel(task, 4)

//     const tasks = []
//     for (let i = 0; i < 10; i++) {
//         tasks.push(f(i))
//     }
//     for (const t of tasks) {
//         console.log(await t)
//     }
// }

main()
