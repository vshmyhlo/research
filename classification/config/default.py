from all_the_tools.config import Config as C

k = 1.

config = C(
    seed=42,
    epochs=100,
    train=C(
        batch_size=int(128 * k),
        opt=C(
            type='sgd',
            lr=0.1 * k,
            momentum=0.9,
            weight_decay=1e-4),
        sched=C(
            type='multistep',
            epochs=[60, 80])),
    eval=C(
        batch_size=int(128 * k)))
