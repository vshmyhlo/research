from all_the_tools.config import Config as C

config = C(
    seed=42,
    epochs=30,
    model=None,
    train=C(
        batch_size=32,
        optimizer=C(
            type='adam',
            lr=1e-3,
            momentum=0.9,
            weight_decay=1e-4,
            lookahead=None),
        scheduler=C(
            type='cosine')),
    eval=C(
        batch_size=1))
