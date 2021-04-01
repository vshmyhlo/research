from all_the_tools.config import Config as C

noise_size = 256

config = C(
    dataset="wikiart",
    num_epochs=10000,
    batches_in_epoch=128,
    batch_size=64,
    image_size=256,
    noise_size=noise_size,
    r1_gamma=10,
    opt=C(
        type="adam",
        args=C(
            lr=0.0025,
            betas=(0.0, 0.99),
            eps=1e-8,
        ),
    ),
    dsc=C(
        loss="logns",
        base_channels=32,
        max_channels=noise_size,
        reg_interval=16,
        batch_std=8,
    ),
    gen=C(
        loss="logns",
        base_channels=32,
        max_channels=noise_size,
        reg_interval=8,
        ema=0.999,
    ),
)
