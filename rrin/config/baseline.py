from all_the_tools.config import Config as C

config = C(
    epochs=50,
    batch_size=10,
    opt=C(
        lr=2.5e-4),
    sched=C(
        steps=[30]))
