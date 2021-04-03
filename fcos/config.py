from yacs.config import CfgNode as CN


def build_default_config():
    config = CN()

    config.seed = 42
    config.dataset = "coco"
    config.epochs = 100
    config.train_batch_size = 32
    config.eval_batch_size = 32
    config.resize_size = 640
    config.crop_size = 640
    config.train_steps = 1000

    config.model = CN()
    config.model.backbone = "resnet50"

    config.anchors = CN()
    config.anchors.min_iou = 0.4
    config.anchors.max_iou = 0.5
    config.anchors.sizes = [
        None,
        None,
        None,
        32,
        64,
        128,
        256,
        512,
    ]
    config.anchors.ratios = [
        0.5,
        1,
        2,
    ]
    config.anchors.scales = [
        1,
        1.26,
        1.587,
    ]

    config.loss = CN()
    config.loss.classification = "focal"
    config.loss.localization = "smooth_l1"

    config.opt = CN()
    config.opt.type = "sgd"
    config.opt.learning_rate = 0.01
    config.opt.weight_decay = 1e-4
    config.opt.acc_steps = 1

    config.opt.sgd = CN()
    config.opt.sgd.momentum = 0.9

    config.sched = CN()
    config.sched.type = "cosine"

    return config
