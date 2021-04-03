from all_the_tools.config import Config as C

config = C(
    seed=42,
    epochs=100,
    model=None,
    train=C(
        batch_size=32,
        optimizer=C(type="sgd", lr=1e-2, momentum=0.9, weight_decay=1e-4),
        scheduler=C(type="cosine"),
    ),
    eval=C(batch_size=1),
)
