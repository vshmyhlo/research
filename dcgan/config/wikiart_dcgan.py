from all_the_tools.config import Config as C

config = C(
    num_epochs=500,
    batch_size=128,
    image_size=256,
    loss="bce",
    opt=C(
        type="adam",
        args=C(
            lr=0.0002,
            betas=(0.5, 0.999),
        ),
    ),
    gen=C(
        base_channels=512,
        kernel_size=4,
    ),
    dsc=C(
        num_steps=1,
        weight_clip=None,
        base_channels=512,
        kernel_size=3,
    ),
)
