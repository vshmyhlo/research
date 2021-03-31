from all_the_tools.config import Config as C

epochs = 20
batch_size = 64
lr = 0.05 * batch_size / 256

# TODO: try "optimal" config


config = C(
    seed=42,
    epochs=epochs,
    image_size=256,
    model=C(
        backbone="resnet34",
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
