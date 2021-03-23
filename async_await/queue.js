class Queue {
    constructor() {
        this.items  = [];
        this.waiting = [];
    }

    async put(item) {
        if (this.waiting.length > 0) {
            const resolve = this.waiting.shift();
            resolve(item);
        } else {
            this.items.push(item);
        }
    }

    async get(item) {
        if (this.items.length > 0) {
            return this.items.shift();
        } else {
            let resolve;
            const p = new Promise(res => {
                resolve = res;
            })
            this.waiting.push(resolve);
            return p;
        }
    }
}

async function sleep(delay) {
    return new Promise(res => {
        setTimeout(res, delay);
    })
}

async function producer(q, count) {
    for (n = 0; n < count; n++) {
        console.log('producing', n);
        await q.put(n);
        await sleep(1000);
    }
    console.log('producer done');
    await q.put(null);
}

async function consumer(q) {
    while (true) {
        const n = await q.get();
        if (n === null) {
            break
        }
        console.log('consuming', n)
    }
    console.log('consumer done');
}

async function case1()  {
    const q = new Queue();
    const c = consumer(q)
    const p = producer(q, 5)

    await c;
    await p;
}

async function case2() {
    const q = new Queue();
    await producer(q, 5);
    await sleep(1);
    await consumer(q);
}

async function main() {
    console.log('case 1');
    await case1();
    console.log('case 2');
    await case2();
}

main()