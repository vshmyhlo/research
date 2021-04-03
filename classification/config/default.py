from all_the_tools.config import Config as C

k = 0.5
epochs = 100
batch_size = 128

config = C(
    seed=42,
    epochs=epochs,
    log_interval=int(epochs * 0.1),
    model="resnet50",
    train=C(
        batch_size=int(batch_size * k),
        opt=C(type="sgd", lr=0.1 * k, momentum=0.9, weight_decay=1e-4),
        sched=C(type="multistep", epochs=[int(epochs * 0.6), int(epochs * 0.8)]),
    ),
    eval=C(batch_size=int(batch_size * k)),
)
