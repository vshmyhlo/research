import os

import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision
import torchvision.transforms as T
from all_the_tools.config import load_config
from all_the_tools.metrics import Last, Mean
from all_the_tools.torch.utils import Saver
from sklearn.model_selection import StratifiedKFold
from tensorboardX import SummaryWriter
from tqdm import tqdm

from fix_match.utils import UDataset, XUDataLoader
from randaugment.transforms import RandAugmentMC
from utils import compute_nrow

NUM_CLASSES = 10
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@click.command()
@click.option("--config-path", type=click.Path(), required=True)
@click.option("--dataset-path", type=click.Path(), required=True)
@click.option("--experiment-path", type=click.Path(), required=True)
@click.option("--restore-path", type=click.Path())
@click.option("--workers", type=click.INT, default=os.cpu_count())
def main(config_path, **kwargs):
    config = load_config(config_path, **kwargs)

    weak_transform, strong_transform, eval_transform = build_transforms()
    x_indices, u_indices = build_x_u_split(
        torchvision.datasets.CIFAR10(config.dataset_path, train=True, download=True),
        config.train.num_labeled,
    )

    x_dataset = torch.utils.data.Subset(
        torchvision.datasets.CIFAR10(
            config.dataset_path, train=True, transform=weak_transform, download=True
        ),
        x_indices,
    )
    u_dataset = UDataset(
        *[
            torch.utils.data.Subset(
                torchvision.datasets.CIFAR10(
                    config.dataset_path, train=True, transform=u_transform, download=True
                ),
                u_indices,
            )
            for u_transform in [weak_transform, strong_transform]
        ]
    )
    eval_dataset = torchvision.datasets.CIFAR10(
        config.dataset_path, train=False, transform=eval_transform, download=True
    )
    train_data_loader = XUDataLoader(
        torch.utils.data.DataLoader(
            x_dataset,
            batch_size=config.train.x_batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=config.workers,
        ),
        torch.utils.data.DataLoader(
            u_dataset,
            batch_size=config.train.u_batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=config.workers,
        ),
    )
    eval_data_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=config.eval.batch_size, num_workers=config.workers
    )

    model = torchvision.models.resnet50(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
    model.apply(weights_init)
    optimizer = build_optimizer(model.parameters(), config)
    scheduler = build_scheduler(optimizer, config, len(train_data_loader))
    saver = Saver(
        {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
        }
    )
    if config.restore_path is not None:
        saver.load(config.restore_path, keys=["model"])

    for epoch in range(1, config.epochs + 1):
        # optimizer.train()
        train_epoch(model, train_data_loader, optimizer, scheduler, epoch=epoch, config=config)
        # optimizer.eval()
        if epoch % config.log_interval != 0:
            continue
        eval_epoch(model, eval_data_loader, epoch=epoch, config=config)
        saver.save(
            os.path.join(config.experiment_path, "checkpoint_{}.pth".format(epoch)), epoch=epoch
        )


def build_x_u_split(dataset, num_labeled):
    targets = torch.tensor([target for _, target in tqdm(dataset, "loading split")])

    ratio = len(dataset) // num_labeled

    u_indices, x_indices = next(
        StratifiedKFold(ratio, shuffle=True, random_state=42).split(targets, targets)
    )
    u_indices, x_indices = torch.tensor(u_indices), torch.tensor(x_indices)

    return x_indices, u_indices


def build_optimizer(parameters, config):
    if config.train.opt.type == "sgd":
        optimizer = torch.optim.SGD(
            parameters,
            config.train.opt.lr,
            momentum=config.train.opt.momentum,
            weight_decay=config.train.opt.weight_decay,
            nesterov=True,
        )
    elif config.train.opt.type == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters,
            config.train.opt.lr,
            momentum=config.train.opt.momentum,
            weight_decay=config.train.opt.weight_decay,
        )
    elif config.train.opt.type == "adam":
        optimizer = torch.optim.Adam(
            parameters, config.train.opt.lr, weight_decay=config.train.opt.weight_decay
        )
    else:
        raise AssertionError("invalid optimizer {}".format(config.train.opt.type))

    # optimizer = EWA(optimizer, momentum=0.999, num_steps=1)

    return optimizer


def build_scheduler(optimizer, config, steps_per_epoch):
    if config.train.sched.type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, config.epochs * steps_per_epoch
        )
    elif config.train.sched.type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=(config.epochs * steps_per_epoch) // 3, gamma=0.1
        )
    elif config.train.sched.type == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [epoch * steps_per_epoch for epoch in config.train.sched.epochs], gamma=0.1
        )
    else:
        raise AssertionError("invalid scheduler {}".format(config.train.sched.type))

    return scheduler


def denormalize(input):
    input = input * 0.25 + 0.5

    return input


def build_transforms():
    to_tensor_and_norm = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.25]),
        ]
    )

    weak_transforms = T.Compose(
        [
            T.RandomHorizontalFlip(),
            T.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode="reflect"),
            to_tensor_and_norm,
        ]
    )
    strong_transform = T.Compose(
        [
            T.RandomHorizontalFlip(),
            T.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode="reflect"),
            RandAugmentMC(n=2, m=10),
            to_tensor_and_norm,
        ]
    )
    eval_transform = T.Compose(
        [
            to_tensor_and_norm,
        ]
    )

    return weak_transforms, strong_transform, eval_transform


def train_epoch(model, data_loader, optimizer, scheduler, epoch, config):
    metrics = {
        "x_loss": Mean(),
        "u_loss": Mean(),
        "u_loss_mask": Mean(),
        "lr": Last(),
    }

    model.train()
    for (x_w_images, x_targets), (u_w_images, u_s_images) in tqdm(
        data_loader, desc="epoch {}/{}, train".format(epoch, config.epochs)
    ):
        x_w_images, x_targets, u_w_images, u_s_images = (
            x_w_images.to(DEVICE),
            x_targets.to(DEVICE),
            u_w_images.to(DEVICE),
            u_s_images.to(DEVICE),
        )

        x_w_logits, u_w_logits, u_s_logits = model(
            torch.cat([x_w_images, u_w_images, u_s_images], 0)
        ).split([x_w_images.size(0), u_w_images.size(0), u_s_images.size(0)])

        # x ############################################################################################################
        x_loss = F.cross_entropy(input=x_w_logits, target=x_targets, reduction="none")
        metrics["x_loss"].update(x_loss.data.cpu().numpy())

        # u ############################################################################################################
        u_loss_mask, u_targets = F.softmax(u_w_logits.detach(), 1).max(1)
        u_loss_mask = (u_loss_mask >= config.train.tau).float()

        u_loss = u_loss_mask * F.cross_entropy(
            input=u_s_logits, target=u_targets, reduction="none"
        )
        metrics["u_loss"].update(u_loss.data.cpu().numpy())
        metrics["u_loss_mask"].update(u_loss_mask.data.cpu().numpy())

        # opt step #####################################################################################################
        metrics["lr"].update(np.squeeze(scheduler.get_lr()))

        optimizer.zero_grad()
        (x_loss.mean() + config.train.u_weight * u_loss.mean()).backward()
        optimizer.step()
        scheduler.step()

    if epoch % config.log_interval != 0:
        return

    writer = SummaryWriter(os.path.join(config.experiment_path, "train"))
    with torch.no_grad():
        for k in metrics:
            writer.add_scalar(k, metrics[k].compute_and_reset(), global_step=epoch)
        writer.add_image(
            "x_w_images",
            torchvision.utils.make_grid(
                denormalize(x_w_images), nrow=compute_nrow(x_w_images), normalize=True
            ),
            global_step=epoch,
        )
        writer.add_image(
            "u_w_images",
            torchvision.utils.make_grid(
                denormalize(u_w_images), nrow=compute_nrow(u_w_images), normalize=True
            ),
            global_step=epoch,
        )
        writer.add_image(
            "u_s_images",
            torchvision.utils.make_grid(
                denormalize(u_s_images), nrow=compute_nrow(u_s_images), normalize=True
            ),
            global_step=epoch,
        )

    writer.flush()
    writer.close()


def eval_epoch(model, data_loader, epoch, config):
    metrics = {
        "x_loss": Mean(),
        "accuracy": Mean(),
    }

    with torch.no_grad():
        model.eval()
        for x_images, x_targets in tqdm(
            data_loader, desc="epoch {}/{}, eval".format(epoch, config.epochs)
        ):
            x_images, x_targets = x_images.to(DEVICE), x_targets.to(DEVICE)

            x_logits = model(x_images)
            x_loss = F.cross_entropy(input=x_logits, target=x_targets, reduction="none")

            metrics["x_loss"].update(x_loss.data.cpu().numpy())
            metrics["accuracy"].update(
                (x_logits.argmax(-1) == x_targets).float().data.cpu().numpy()
            )

    writer = SummaryWriter(os.path.join(config.experiment_path, "eval"))
    with torch.no_grad():
        for k in metrics:
            writer.add_scalar(k, metrics[k].compute_and_reset(), global_step=epoch)
        writer.add_image(
            "x_images",
            torchvision.utils.make_grid(
                denormalize(x_images), nrow=compute_nrow(x_images), normalize=True
            ),
            global_step=epoch,
        )

    writer.flush()
    writer.close()


def weights_init(m):
    if isinstance(m, (nn.Conv2d,)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d,)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


if __name__ == "__main__":
    main()
