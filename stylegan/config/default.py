from all_the_tools.config import Config as C

noise_size = 512

config = C(
    num_epochs=10000,
    batches_in_epoch=256,
    batch_size=32,
    image_size=1024,
    noise_size=noise_size,
    r1_gamma=10,
    opt=C(
        type="adam",
        args=C(
            lr=0.002,
            betas=(0.0, 0.99),
            eps=1e-8,
        ),
    ),
    dsc=C(
        loss="logns",
        base_channels=32,
        max_channels=noise_size,
        reg_interval=16,
    ),
    gen=C(
        loss="logns",
        base_channels=32,
        max_channels=noise_size,
    ),
)
