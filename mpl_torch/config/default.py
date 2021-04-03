from all_the_tools.config import Config as C

batch_size = 128
epochs = 1000

config = C(
    seed=42,
    epochs=epochs,
    log_interval=max(epochs // 200, 1),
    train=C(
        num_labeled=4000,
        x_batch_size=batch_size,
        u_batch_size=batch_size,
        student=C(dropout=0.35, opt=C(type="sgd", lr=0.3, momentum=0.9, weight_decay=5e-4)),
        teacher=C(dropout=0.5, opt=C(type="sgd", lr=0.125, momentum=0.9, weight_decay=5e-4)),
        sched=C(type="warmup_cosine"),
    ),
    eval=C(batch_size=batch_size),
)
