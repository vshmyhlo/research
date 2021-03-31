from all_the_tools.config import Config as C

config = C(
    num_epochs=10000,
    batch_size=32,
    image_size=128,
    noise_size=256,
    opt=C(
        type="adam",
        args=C(
            lr=0.001,
            betas=(0.0, 0.99),
            eps=1e-8,
        ),
    ),
    dsc=C(
        loss="sp",
        num_steps=1,
        weight_clip=None,
        base_channels=16,
    ),
    gen=C(
        loss="sp",
        base_channels=16,
    ),
)
