from all_the_tools.config import Config as C

# TODO: try "optimal" config
epochs = 100
batch_size = 32
lr = 0.05 * batch_size / 256


config = C(
    seed=42,
    epochs=epochs,
    image_size=384,
    model=C(
        backbone="resnet50",
    ),
    train=C(
        batch_size=batch_size,
        random_resize_scale=2 / 3,
        opt=C(
            type="sgd",
            lr=lr,
            weight_decay=1e-4,
            look_ahead=None,
            sgd=C(
                momentum=0.9,
            ),
            ada_belief=C(
                weight_decouple=False,
            ),
        ),
        sched=C(
            type="cosine",
            multistep=C(
                steps=[round(epochs * 0.6), round(epochs * 0.8)],
            ),
        ),
    ),
)
