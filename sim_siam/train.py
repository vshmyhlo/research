import os

import click
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision
import torchvision.transforms as T
from adabelief_pytorch import AdaBelief
from all_the_tools.config import load_config
from all_the_tools.metrics import Last
from all_the_tools.torch.metrics import Mean
from all_the_tools.torch.optim import LookAhead
from tensorboardX import SummaryWriter
from tqdm import tqdm

from classification.transforms import random_resize
from sim_siam.dataset import DualViewDataset
from sim_siam.model import Model
from utils import MEAN, STD, ChunkedDataLoader, worker_init_fn

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# TODO: lr scheduler account for limited data loader
# TODO: add gauss blur
# TODO: distort aspect ratio
# TODO: random crop size
# TODO: workers random init
# TODO: source train/val from metabase query
# TODO: crop doc using bounding box and not min/max
# TODO: fix precision/recall
# TODO: do not compute loss if no positive classes
# TODO: play with ransac
# TODO: focal loss
# TODO: IoU loss is defined for non-empty classes
# TODO: focal loss and resulting thresholds


@click.command()
@click.option("--config-path", type=click.Path(), required=True)
@click.option("--dataset-path", type=click.Path(), required=True)
@click.option("--experiment-path", type=click.Path(), required=True)
@click.option("--workers", type=click.INT, default=os.cpu_count())
def main(config_path, **kwargs):
    config = load_config(config_path, **kwargs)

    train_transform = build_transforms(config)

    train_data_loader = torch.utils.data.DataLoader(
        DualViewDataset(config.dataset_path, transform=train_transform),
        batch_size=config.train.batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=config.workers,
        worker_init_fn=worker_init_fn,
    )
    train_data_loader = ChunkedDataLoader(
        train_data_loader,
        size=round(len(train_data_loader) / (config.epochs / 1)),
    )

    model = Model(backbone=config.model.backbone).to(DEVICE)
    optimizer = build_optimizer(model.parameters(), config)
    scheduler = build_scheduler(optimizer, config, len(train_data_loader))

    best_score = -float("inf")
    for epoch in range(1, config.epochs + 1):
        train_epoch(
            model,
            train_data_loader,
            optimizer,
            scheduler,
            epoch=epoch,
            config=config,
        )
        score = epoch

        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model": model.backbone.state_dict(),
                    "config": config,
                },
                os.path.join(config.experiment_path, "checkpoint.pth"),
            )
            print("new best score saved: {:.4f}".format(score))


def build_optimizer(parameters, config):
    if config.train.opt.type == "sgd":
        optimizer = torch.optim.SGD(
            parameters,
            lr=config.train.opt.lr,
            momentum=config.train.opt.sgd.momentum,
            weight_decay=config.train.opt.weight_decay,
            nesterov=True,
        )
    elif config.train.opt.type == "adam":
        optimizer = torch.optim.Adam(
            parameters,
            lr=config.train.opt.lr,
            weight_decay=config.train.opt.weight_decay,
        )
    elif config.train.opt.type == "ada_belief":
        optimizer = AdaBelief(
            parameters,
            lr=config.train.opt.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=config.train.opt.weight_decay,
            weight_decouple=config.train.opt.ada_belief.weight_decouple,
            rectify=False,
            fixed_decay=False,
            amsgrad=False,
        )
    else:
        raise AssertionError("invalid optimizer {}".format(config.train.opt.type))

    if config.train.opt.look_ahead is not None:
        optimizer = LookAhead(
            optimizer,
            lr=config.train.opt.look_ahead.lr,
            num_steps=config.train.opt.look_ahead.num_steps,
        )

    return optimizer


def build_scheduler(optimizer, config, steps_per_epoch):
    if config.train.sched.type == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[epoch * steps_per_epoch for epoch in config.train.sched.multisteps.steps],
            gamma=0.1,
        )
    elif config.train.sched.type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs * steps_per_epoch,
        )
    else:
        raise AssertionError("invalid scheduler {}".format(config.train.sched.type))

    return scheduler


def build_transforms(config):
    image_size = round(config.image_size / 1.4), config.image_size

    train_transform = T.Compose(
        [
            # T.RandomChoice(
            #     [
            #         T.RandomAffine(degrees=10, shear=0),
            #         T.RandomAffine(degrees=0, shear=10),
            #     ]
            # ),
            random_resize(image_size, scale=config.train.random_resize_scale),
            T.RandomCrop(image_size, pad_if_needed=True),
            T.RandomApply(
                [T.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                p=0.8,
            ),
            T.RandomApply(
                [T.Grayscale(3)],
                p=0.2,
            ),
            # T.GaussianBlur(..., sigma=[0.1, 2.0]),
            T.ToTensor(),
            # T.RandomApply(
            #     [Scale()],
            #     p=0.2,
            # ),
            T.Normalize(MEAN, STD),
        ]
    )

    return train_transform


def train_epoch(model, data_loader, optimizer, scheduler, epoch, config):
    metrics = {
        "loss": Mean(),
        "lr": Last(),
        "p/norm": Mean(),
        "z/norm": Mean(),
        "p/std": Mean(),
        "z/std": Mean(),
    }

    model.train()
    for images1, images2 in tqdm(
        data_loader,
        desc="epoch {}/{}, train".format(epoch, config.epochs),
    ):
        images1, images2 = images1.to(DEVICE), images2.to(DEVICE)

        p1, z1 = model(images1)
        p2, z2 = model(images2)

        loss = (
            sum(
                [
                    compute_loss(p=p1, z=z2.detach()),
                    compute_loss(p=p2, z=z1.detach()),
                ]
            )
            / 2
        )

        metrics["loss"].update(loss.detach())
        metrics["lr"].update(np.squeeze(scheduler.get_last_lr()))

        for k, v in [("p", p1), ("p", p2), ("z", z1), ("z", z2)]:
            v = v.detach()
            metrics["{}/norm".format(k)].update(v.norm(dim=1))
            v = F.normalize(v, dim=1)
            metrics["{}/std".format(k)].update(v.std(dim=0))

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
        scheduler.step()

    writer = SummaryWriter(os.path.join(config.experiment_path, "train"))
    with torch.no_grad():
        images = torch.cat([images1, images2], 3)

        metrics = {k: metrics[k].compute_and_reset() for k in metrics}
        for k in metrics:
            writer.add_scalar(k, metrics[k], global_step=epoch)
        writer.add_image(
            "images",
            torchvision.utils.make_grid(images[:16], nrow=1, normalize=True),
            global_step=epoch,
        )

    writer.flush()
    writer.close()


def compute_loss(p, z):
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    loss = -(p * z).sum(dim=1)

    return loss


if __name__ == "__main__":
    main()
