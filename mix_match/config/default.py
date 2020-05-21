from all_the_tools.config import Config as C

k = 0.5
epochs = 1000
batch_size = 128

config = C(
    seed=42,
    epochs=epochs,
    epochs_warmup=int(epochs * 0.1),
    log_interval=int(epochs * 0.01),
    model='resnet50',
    train=C(
        batch_size=int(batch_size * k),
        num_labeled=4000,
        mix_match=C(
            weight_u=1.,
            temp=0.5,
            alpha=0.75),
        opt=C(
            type='sgd',
            lr=0.1 * k,
            momentum=0.9,
            weight_decay=1e-4),
        sched=C(
            type='warmup_cosine')),
    eval=C(
        batch_size=int(batch_size * k)))
