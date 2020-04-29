from all_the_tools.config import Config as C

config = C(
    epochs=100,
    batch_size=6,
    opt=C(
        lr=1.5e-4),
    sched=C(
        steps=[60]))
