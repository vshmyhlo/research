from all_the_tools.config import Config as C

config = C(
    epochs=20,
    dataset='mnist',
    image_size=32,
    batch_size=128,
    latent_size=128,
    model=C(
        base_features=16),
    opt=C(
        lr=2e-4))
