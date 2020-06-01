from all_the_tools.config import Config as C

k = 2.
epochs = 30
batch_size = 32

config = C(
    seed=42,
    image_size=256,
    crop_size=224,
    model='resnet50',
    train=C(
        epochs=epochs,
        batch_size=int(batch_size * k),
        opt=C(
            type='sgd',
            lr=0.1 * k,
            momentum=0.9,
            weight_decay=1e-4),
        sched=C(
            type='warmup_cosine',
            epochs_warmup=1)),
    eval=C(
        batch_size=int(batch_size * k)))
