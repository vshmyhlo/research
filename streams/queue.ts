class Queue<T> {
    items: T[]
    waiting: ((x: T) => void)[]

    constructor() {
        this.items = []
        this.waiting = []
    }

    async put(item: T) {
        if (this.waiting.length > 0) {
            const resolve = this.waiting.shift()
            resolve(item)
        } else {
            this.items.push(item)
        }
    }

    async get(): Promise<T> {
        if (this.items.length > 0) {
            return this.items.shift()
        } else {
            let resolve
            const p = new Promise<T>((res) => {
                resolve = res
            })
            this.waiting.push(resolve)
            return p
        }
    }
}

export default Queue
