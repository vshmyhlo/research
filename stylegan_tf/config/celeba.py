from all_the_tools.config import Config as C

config = C(
    epochs=100,
    dataset='celeba',
    image_size=64,
    batch_size=8**2,
    latent_size=128,
    opt=C(
        lr=1e-3,
        beta=(0., 0.99)))
