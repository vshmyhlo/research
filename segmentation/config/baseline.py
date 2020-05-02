from config import Config as C

config = C(
    seed=42,
    epochs=10,
    model=None,
    train=C(
        batch_size=32,
        optimizer=C(
            type='sgd',
            lr=0.01,
            momentum=0.9,
            weight_decay=1e-4,
            lookahead=None),
        scheduler=C(
            type='cosine')),
    eval=C(
        batch_size=32))
