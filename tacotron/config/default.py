from all_the_tools.config import Config as C

config = C(
    seed=42,
    sample_rate=22050,
    train=C(
        batch_size=32))
