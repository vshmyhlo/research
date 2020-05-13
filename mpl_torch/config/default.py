from all_the_tools.config import Config as C

config = C(
    seed=42,
    epochs=1000,
    log_interval=1000 // 50,
    train=C(
        num_labeled=250,
        x_batch_size=32,
        u_batch_size=32 * 1,
        opt=C(
            type='sgd',
            lr=0.03,
            momentum=0.9,
            weight_decay=5e-4),
        sched=C(
            type='cosine')),
    eval=C(
        batch_size=32))
