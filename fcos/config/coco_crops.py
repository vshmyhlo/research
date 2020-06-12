from all_the_tools.config import Config as C

batch_size = 8
config = C(
    seed=42,
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
        batch_size=batch_size),
    eval=C(
        batch_size=batch_size * 2))
