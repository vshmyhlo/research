from all_the_tools.config import Config as C

config = C(
    epochs=8 * 6,
    dataset='celeba',
    image_size=128,
    batch_size=64,
    latent_size=128,
    opt=C(
        lr=1e-3))
