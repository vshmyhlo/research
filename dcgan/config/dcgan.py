from all_the_tools.config import Config as C

config = C(
    num_epochs=100,
    batch_size=128,
    loss="bce",
    opt=C(
        type="adam",
        args=C(
            lr=0.0002,
            betas=(0.5, 0.999),
        ),
    ),
    dsc=C(
        num_steps=1,
        clip=None,
    ),
)
