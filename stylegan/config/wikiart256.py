from all_the_tools.config import Config as C

noise_size = 256

config = C(
    dataset="wikiart",
    num_epochs=10000,
    batches_in_epoch=128,
    batch_size=32,
    image_size=256,
    noise_size=noise_size,
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
        r1_gamma=10,
        batch_std=8,
    ),
    gen=C(
        loss="logns",
        base_channels=32,
        max_channels=noise_size,
        reg_interval=8,
        pl_decay=0.01,
        pl_weight=1,  # TODO:
        ema=0.999,
    ),
)
