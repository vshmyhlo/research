from all_the_tools.config import Config as C

config = C(
    num_epochs=100,
    batch_size=64,
    loss="wass",
    opt=C(
        type="rmsprop",
        args=C(
            lr=0.00005,
        ),
    ),
    dsc=C(
        num_steps=5,
        weight_clip=0.01,
    ),
)
