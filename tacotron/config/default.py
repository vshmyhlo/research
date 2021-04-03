from all_the_tools.config import Config as C

epochs = 50

config = C(
    seed=42,
    sample_rate=22050,
    model=C(num_mels=80, base_features=256),
    train=C(
        epochs=epochs,
        batch_size=32,
        clip_grad_norm=1.0,
        opt=C(type="adam", lr=1e-3, beta=(0.9, 0.999), eps=1e-6, weight_decay=1e-6),
        sched=C(type="warmup_cosine", epochs_warmup=0),
    ),
    eval=C(batch_size=32),
)
