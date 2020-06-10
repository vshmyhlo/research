from all_the_tools.config import Config as C

epochs = 20
batch_size = 64

config = C(
    seed=42,
    image_size=256,
    crop_size=256,
    model='effnet-b0',
    train=C(
        epochs=epochs,
        batch_size=batch_size,
        loss=['ce'],
        opt=C(
            type='adam',
            lr=1e-3,
            momentum=0.9,
            weight_decay=1e-4,
            la=C(
                lr=0.5,
                steps=5),
            ema=C(
                mom=0.96,
                steps=5)),
        sched=C(
            type='warmup_cosine',
            epochs_warmup=0)),
    eval=C(
        batch_size=batch_size))
