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
from tensorboardX import SummaryWriter
from tqdm import tqdm

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

    train_transform, eval_transform = build_transforms()

    train_data_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(config.dataset_path, train=True, transform=train_transform),
        batch_size=config.train.batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=config.workers,
    )
    eval_data_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(config.dataset_path, train=False, transform=eval_transform),
        batch_size=config.eval.batch_size,
        num_workers=config.workers,
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
        train_epoch(model, train_data_loader, optimizer, scheduler, epoch=epoch, config=config)
        eval_epoch(model, eval_data_loader, epoch=epoch, config=config)
        saver.save(
            os.path.join(config.experiment_path, "checkpoint_{}.pth".format(epoch)), epoch=epoch
        )


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
    def assert_shape(input):
        assert input.size() == (3, 32, 32)

        return input

    to_tensor_and_norm = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.25]),
            assert_shape,
        ]
    )
    train_transform = T.Compose(
        [
            T.RandomHorizontalFlip(),
            T.RandomCrop(size=32, padding=4, padding_mode="reflect"),
            T.ColorJitter(0.3, 0.3, 0.3),
            to_tensor_and_norm,
            T.RandomErasing(),
        ]
    )
    eval_transform = T.Compose(
        [
            to_tensor_and_norm,
        ]
    )

    return train_transform, eval_transform


def train_epoch(model, data_loader, optimizer, scheduler, epoch, config):
    metrics = {
        "loss": Mean(),
        "lr": Last(),
    }

    model.train()
    for images, targets in tqdm(
        data_loader, desc="epoch {}/{}, train".format(epoch, config.epochs)
    ):
        images, targets = images.to(DEVICE), targets.to(DEVICE)

        logits = model(images)
        loss = F.cross_entropy(input=logits, target=targets, reduction="none")

        metrics["loss"].update(loss.data.cpu().numpy())
        metrics["lr"].update(np.squeeze(scheduler.get_lr()))

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
        scheduler.step()

    writer = SummaryWriter(os.path.join(config.experiment_path, "train"))
    with torch.no_grad():
        for k in metrics:
            writer.add_scalar(k, metrics[k].compute_and_reset(), global_step=epoch)
        writer.add_image(
            "images",
            torchvision.utils.make_grid(
                denormalize(images), nrow=compute_nrow(images), normalize=True
            ),
            global_step=epoch,
        )
        writer.add_histogram("params", flatten_weights(model.parameters()), global_step=epoch)

    writer.flush()
    writer.close()


def eval_epoch(model, data_loader, epoch, config):
    metrics = {
        "loss": Mean(),
        "accuracy": Mean(),
    }

    with torch.no_grad():
        model.eval()
        for images, targets in tqdm(
            data_loader, desc="epoch {}/{}, eval".format(epoch, config.epochs)
        ):
            images, targets = images.to(DEVICE), targets.to(DEVICE)

            logits = model(images)
            loss = F.cross_entropy(input=logits, target=targets, reduction="none")

            metrics["loss"].update(loss.data.cpu().numpy())
            metrics["accuracy"].update((logits.argmax(-1) == targets).float().data.cpu().numpy())

    writer = SummaryWriter(os.path.join(config.experiment_path, "eval"))
    with torch.no_grad():
        for k in metrics:
            writer.add_scalar(k, metrics[k].compute_and_reset(), global_step=epoch)
        writer.add_image(
            "images",
            torchvision.utils.make_grid(
                denormalize(images), nrow=compute_nrow(images), normalize=True
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


def flatten_weights(params):
    params = torch.cat([p.data.view(-1) for p in params if p.grad is not None])

    return params


if __name__ == "__main__":
    main()
