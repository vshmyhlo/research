import os

import click
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as T
from all_the_tools.config import load_config
from all_the_tools.metrics import Last, Mean
from all_the_tools.torch.optim import EMA, LookAhead
from all_the_tools.torch.utils import Saver
from tensorboardX import SummaryWriter
from tqdm import tqdm

from losses import sigmoid_cross_entropy
from mela.dataset import Dataset
from mela.model import Model
from mela.sampler import BatchSampler
from mela.transforms import LoadImage, RandomResizedCrop
from mela.utils import Concat
from scheduler import WarmupCosineAnnealingLR
from transforms import ApplyTo, Extract
from utils import compute_nrow, random_seed

# TODO: no sub-crop
# TODO: compute stats
# TODO: very large lr
# TODO: pick sharpening T
# TODO: preds hist
# TODO: better eda
# TODO: add loss to wide sigmoid values away
# TODO: spat trans net
# TODO: ten-crop method
# TODO: https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/154876
# TODO: aspect ratio distortion
# TODO: progressive resize
# TODO: fix seed
# TODO: lookahead, ewa and friends
# TODO: onecycle?
# TODO: use metainfo
# TODO: mosaic aug
# TODO: https://towardsdatascience.com/explicit-auc-maximization-70beef6db14e
# TODO: lsep loss
# TODO: focal loss
# TODO: soft f1 loss
# TODO: lookahead
# TODO: aug
# TODO: mixup/cutmix


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MEAN = torch.tensor([0.4914, 0.4822, 0.4465])
STD = torch.tensor([0.2470, 0.2435, 0.2616])


@click.command()
@click.option('--config-path', type=click.Path(), required=True)
@click.option('--dataset-path', type=click.Path(), required=True)
@click.option('--experiment-path', type=click.Path(), required=True)
@click.option('--fold', type=click.INT, required=True)
@click.option('--restore-path', type=click.Path())
@click.option('--workers', type=click.INT, default=os.cpu_count())
def main(config_path, **kwargs):
    config = load_config(
        config_path,
        **kwargs)
    config.experiment_path = os.path.join(config.experiment_path, 'F{}'.format(config.fold))
    del kwargs
    random_seed(config.seed)

    train_transform, eval_transform = build_transforms(config)

    train_dataset = Dataset(config.dataset_path, train=True, fold=config.fold, transform=train_transform)
    eval_dataset = Dataset(config.dataset_path, train=False, fold=config.fold, transform=eval_transform)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=BatchSampler(
            train_dataset.data['target'], shuffle=True, drop_last=True),
        num_workers=config.workers)
    eval_data_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_sampler=BatchSampler(
            eval_dataset.data['target'], shuffle=False, drop_last=False),
        num_workers=config.workers)

    model = Model(config.model).to(DEVICE)
    optimizer = build_optimizer(model.parameters(), config)
    scheduler = build_scheduler(optimizer, config, len(train_data_loader))
    saver = Saver({
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
    })
    if config.restore_path is not None:
        saver.load(config.restore_path, keys=['model'])

    for epoch in range(1, config.train.epochs + 1):
        optimizer.train()
        train_epoch(model, train_data_loader, optimizer, scheduler, epoch=epoch, config=config)
        eval_epoch(model, eval_data_loader, epoch=epoch, config=config)
        optimizer.eval()
        eval_epoch(model, eval_data_loader, epoch=epoch, config=config, suffix='ema')
        saver.save(
            os.path.join(config.experiment_path, 'checkpoint_{}.pth'.format(epoch)),
            epoch=epoch)


def build_optimizer(parameters, config):
    if config.train.opt.type == 'sgd':
        optimizer = torch.optim.SGD(
            parameters,
            config.train.opt.lr,
            momentum=config.train.opt.momentum,
            weight_decay=config.train.opt.weight_decay,
            nesterov=True)
    elif config.train.opt.type == 'adam':
        optimizer = torch.optim.Adam(
            parameters,
            config.train.opt.lr,
            weight_decay=config.train.opt.weight_decay)
    else:
        raise AssertionError('invalid optimizer {}'.format(config.train.opt.type))

    optimizer = LookAhead(optimizer, lr=0.5, num_steps=5)
    optimizer = EMA(optimizer, momentum=config.train.opt.ema, num_steps=1)

    return optimizer


def build_scheduler(optimizer, config, steps_per_epoch):
    if config.train.sched.type == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            [epoch * steps_per_epoch for epoch in config.train.sched.epochs],
            gamma=0.1)
    elif config.train.sched.type == 'warmup_cosine':
        scheduler = WarmupCosineAnnealingLR(
            optimizer,
            epoch_warmup=config.train.sched.epochs_warmup * steps_per_epoch,
            epoch_max=config.train.epochs * steps_per_epoch)
    else:
        raise AssertionError('invalid scheduler {}'.format(config.train.sched.type))
    return scheduler


def build_transforms(config):
    train_transform = T.Compose([
        LoadImage(T.Resize(config.image_size)),
        ApplyTo(
            'image',
            T.Compose([
                RandomResizedCrop(config.crop_size),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ColorJitter(0.1, 0.1, 0.1),
                # ColorConstancy(),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD),
            ])),
        Extract(['image', 'meta', 'target']),
    ])
    eval_transform = T.Compose([
        LoadImage(T.Resize(config.image_size)),
        ApplyTo(
            'image',
            T.Compose([
                T.CenterCrop(config.crop_size),
                # ColorConstancy(),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD),
            ])),
        Extract(['image', 'meta', 'target']),
    ])

    return train_transform, eval_transform


def train_epoch(model, data_loader, optimizer, scheduler, epoch, config):
    metrics = {
        'loss': Mean(),
        'lr': Last(),
    }

    model.train()
    for images, meta, targets in \
            tqdm(data_loader, desc='fold {}, epoch {}/{}, train'.format(config.fold, epoch, config.train.epochs)):
        images, meta, targets = images.to(DEVICE), {k: meta[k].to(DEVICE) for k in meta}, targets.to(DEVICE)
        # images, targets = cut_mix(images, targets, alpha=1.)

        logits = model(images, meta)
        loss = compute_loss(input=logits, target=targets)

        metrics['loss'].update(loss.data.cpu().numpy())
        metrics['lr'].update(np.squeeze(scheduler.get_last_lr()))

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
        scheduler.step()

    metrics = {k: metrics[k].compute_and_reset() for k in metrics}
    writer = SummaryWriter(os.path.join(config.experiment_path, 'train'))
    with torch.no_grad():
        for k in metrics:
            writer.add_scalar(k, metrics[k], global_step=epoch)
        writer.add_image('images', torchvision.utils.make_grid(
            images, nrow=compute_nrow(images), normalize=True), global_step=epoch)

    writer.flush()
    writer.close()


def eval_epoch(model, data_loader, epoch, config, suffix=''):
    metrics = {
        'loss': Mean(),
    }
    all_targets = Concat()
    all_logits = Concat()

    with torch.no_grad():
        model.eval()
        for images, meta, targets in \
                tqdm(data_loader, desc='fold {}, epoch {}/{}, eval'.format(config.fold, epoch, config.train.epochs)):
            images, meta, targets = images.to(DEVICE), {k: meta[k].to(DEVICE) for k in meta}, targets.to(DEVICE)

            logits = model(images, meta)
            loss = compute_loss(input=logits, target=targets)

            metrics['loss'].update(loss.data.cpu().numpy())

            all_targets.update(targets.cpu())
            all_logits.update(logits.cpu())

    all_targets = all_targets.compute()
    all_logits = all_logits.compute()

    metrics = {
        **{k: metrics[k].compute_and_reset() for k in metrics},
        **compute_metric(input=all_logits, target=all_targets),
    }
    roc_curve = plot_roc_curve(input=all_logits, target=all_targets)
    writer = SummaryWriter(os.path.join(config.experiment_path, 'eval', suffix))
    with torch.no_grad():
        for k in metrics:
            writer.add_scalar(k, metrics[k], global_step=epoch)
        writer.add_image('images', torchvision.utils.make_grid(
            images, nrow=compute_nrow(images), normalize=True), global_step=epoch)
        writer.add_figure('roc_curve', roc_curve, global_step=epoch)

    writer.flush()
    writer.close()


def compute_loss(input, target):
    loss = [
        sigmoid_cross_entropy(input=input, target=target),
        # lsep_loss(input=input, target=target),
        # sigmoid_focal_loss(input=input, target=target),
        # f1_loss(input=input.sigmoid(), target=target),
    ]

    loss = sum(x.mean() for x in loss)

    return loss


def compute_metric(input, target):
    roc_auc = sklearn.metrics.roc_auc_score(
        y_score=input.sigmoid().data.cpu().numpy(),
        y_true=target.data.cpu().numpy())

    return {
        'roc_auc': roc_auc,
    }


def plot_roc_curve(input, target):
    fpr, tpr, _ = sklearn.metrics.roc_curve(
        y_score=input.sigmoid().data.cpu().numpy(),
        y_true=target.data.cpu().numpy())

    fig = plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.fill_between(fpr, 0, tpr, alpha=0.1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    return fig


if __name__ == '__main__':
    main()