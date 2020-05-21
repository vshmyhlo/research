from all_the_tools.config import Config as C

epochs = 1000
batch_size = 32

# 16,000

config = C(
    seed=42,
    epochs=epochs,
    log_interval=epochs // 100,
    train=C(
        num_labeled=4000,
        batch_size=batch_size,
        weight_u=75.,
        temp=0.5,
        alpha=0.75,
        opt=C(
            type='sgd',
            lr=0.03,
            momentum=0.9,
            weight_decay=1e-4),
        sched=C(
            type='cosine')),
    eval=C(
        batch_size=batch_size))
