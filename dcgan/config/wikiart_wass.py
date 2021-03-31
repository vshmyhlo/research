from all_the_tools.config import Config as C

config = C(
    num_epochs=500,
    batch_size=64,
    image_size=256,
    loss="wass",
    opt=C(
        type="rmsprop",
        args=C(
            lr=0.00005,
        ),
    ),
    gen=C(
        base_channels=512,
        kernel_size=4,
    ),
    dsc=C(
        num_steps=5,
        weight_clip=0.01,
        base_channels=512,
        kernel_size=3,
    ),
)
