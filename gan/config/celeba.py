from all_the_tools.config import Config as C

config = C(
    epochs=30,
    dataset='celeba',
    image_size=128,
    batch_size=128,
    latent_size=128,
    model=C(
        base_features=16),
    opt=C(
        lr=2e-4))
