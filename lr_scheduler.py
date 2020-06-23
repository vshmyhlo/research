import numpy as np
import torch.optim


class WarmupCosineAnnealingLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, epoch_warmup, epoch_max, last_epoch=-1):
        def f(epoch):
            if epoch < epoch_warmup:
                return epoch / epoch_warmup
            else:
                return (np.cos((epoch - epoch_warmup) / (epoch_max - epoch_warmup) * np.pi) + 1) / 2

        super().__init__(optimizer, f, last_epoch=last_epoch)
