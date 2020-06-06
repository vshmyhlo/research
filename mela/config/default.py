from all_the_tools.config import Config as C

epochs = 20
batch_size = 54

config = C(
    seed=42,
    image_size=256,
    crop_size=224,
    model='effnet-b0',
    train=C(
        epochs=epochs,
        batch_size='balanced',
        loss=['ce', 'lsep'],
        opt=C(
            type='adam',
            lr=1e-3,
            momentum=0.9,
            weight_decay=1e-4,
            ema=0.995),
        sched=C(
            type='warmup_cosine',
            epochs_warmup=0)),
    eval=C(
        batch_size=32))
