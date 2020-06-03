from all_the_tools.config import Config as C

epochs = 30
batch_size = 54

config = C(
    seed=42,
    image_size=256,
    crop_size=224,
    model='resnet50',
    train=C(
        epochs=epochs,
        batch_size=None,
        opt=C(
            type='sgd',
            lr=0.1,
            momentum=0.9,
            weight_decay=1e-4,
            ema=0.995),
        sched=C(
            type='warmup_cosine',
            epochs_warmup=0)),
    eval=C(
        batch_size=None))
