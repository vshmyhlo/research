from all_the_tools.config import Config as C

batch_size = 16

config = C(
    seed=42,
    train_steps=1000,
    resize_size=512,
    crop_size=512,
    dataset='coco',
    model=C(
        backbone='resnet50',
        levels=[
            None,
            None,
            None,
            (0, 64),
            (64, 128),
            (128, 256),
            (256, 512),
            (512, float('inf')),
        ]),
    train=C(
        epochs=90,
        batch_size=batch_size,
        acc_steps=1,
        opt=C(
            type='sgd',
            learning_rate=0.01,
            weight_decay=1e-4,
            momentum=0.9),

        sched=C(
            type='step',
            steps=[60, 80])),
    eval=C(
        batch_size=batch_size * 2))
