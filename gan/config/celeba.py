from all_the_tools.config import Config as C

config = C(
    epochs=8 * 5,
    dataset="celeba",
    image_size=128,
    batch_size=64,
    latent_size=128,
    grow_min_level=1,
    opt=C(lr=1e-3, beta=(0.0, 0.99)),
)
