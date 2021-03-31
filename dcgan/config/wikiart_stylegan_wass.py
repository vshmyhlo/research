from all_the_tools.config import Config as C

config = C(
    num_epochs=500,
    batch_size=32,
    image_size=128,
    noise_size=256,
    opt=C(
        type="adam",
        args=C(
            lr=0.0002,
            betas=(0.0, 0.99),
            eps=1e-8,
        ),
    ),
    dsc=C(
        loss="wass",
        num_steps=5,
        weight_clip=0.01,
        base_channels=16,
    ),
    gen=C(
        loss="wass",
        base_channels=16,
    ),
)
