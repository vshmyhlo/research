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
from all_the_tools.metrics import Last
from all_the_tools.torch.optim import EMA, LookAhead
from all_the_tools.torch.utils import Saver
from tensorboardX import SummaryWriter
from tqdm import tqdm

from losses import lsep_loss, f1_loss, sigmoid_cross_entropy
from mela.dataset import Dataset2020KFold, ConcatDataset
from mela.model import Model
from mela.transforms import LoadImage
from mela.utils import Concat, Mean
from scheduler import WarmupCosineAnnealingLR
from transforms import ApplyTo, Extract
from transforms.image import Random8
from utils import compute_nrow, random_seed

# TODO: average best cps based on their score
# TODO: TTA / ten-crop
# TODO: segmentation
# TODO: dropcut
# TODO: scheduler app
# TODO: more color jitter
# TODO: build model based on deltas with adjacent pixels
# TODO: error analysis
# TODO: semi-sup, self-sup
# TODO: larger net, larger crop
# TODO: focal loss
# TODO: mixup
# TODO: pseudolabeling
# TODO: external data
# TODO: eval with tta


# TODO: compute stats
# TODO: probs hist
# TODO: double batch size
# TODO: better eda
# TODO: spat trans net
# TODO: focal loss
# TODO: copy config to exp-path
# TODO: https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/154876
# TODO: check extarnal data intersection
# TODO: check external meta
# TODO: merge external targets
# TODO: aspect ratio distortion
# TODO: use metainfo
# TODO: mosaic aug
# TODO: https://towardsdatascience.com/explicit-auc-maximization-70beef6db14e
# TODO: predict other classes
# TODO: focal loss after sampler fix
# TODO: shear, rotate, other augs from torch transforms

# TODO: mixup/cutmix after sampler fix
# TODO: drop unknown
# TODO: progressive resize


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

    train_dataset = ConcatDataset([
        Dataset2020KFold(
            os.path.join(config.dataset_path, '2020'), train=True, fold=config.fold, transform=train_transform),
        # Dataset2019(
        #     os.path.join(config.dataset_path, '2019'), transform=train_transform),
    ])
    eval_dataset = Dataset2020KFold(
        os.path.join(config.dataset_path, '2020'), train=False, fold=config.fold, transform=eval_transform)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.workers)
    eval_data_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=config.eval.batch_size,
        shuffle=False,
        drop_last=False,
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

    best_score = 0.
    for epoch in range(1, config.train.epochs + 1):
        optimizer.train()
        train_epoch(model, train_data_loader, optimizer, scheduler, epoch=epoch, config=config)

        score = eval_epoch(model, eval_data_loader, epoch=epoch, config=config)
        # saver.save(os.path.join(config.experiment_path, 'eval', 'checkpoint_{}.pth'.format(epoch)), epoch=epoch)
        if score > best_score:
            best_score = score
            saver.save(os.path.join(config.experiment_path, 'checkpoint_best.pth'.format(epoch)), epoch=epoch)

        optimizer.eval()
        score = eval_epoch(model, eval_data_loader, epoch=epoch, config=config, suffix='ema')
        # saver.save(os.path.join(config.experiment_path, 'eval', 'ema', 'checkpoint_{}.pth'.format(epoch)), epoch=epoch)
        if score > best_score:
            best_score = score
            saver.save(os.path.join(config.experiment_path, 'checkpoint_best.pth'.format(epoch)), epoch=epoch)


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

    optimizer = LookAhead(optimizer, lr=config.train.opt.la.lr, num_steps=config.train.opt.la.steps)
    optimizer = EMA(optimizer, momentum=config.train.opt.ema.mom, num_steps=config.train.opt.ema.steps)

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
                T.RandomCrop(config.crop_size),
                Random8(),
                T.ColorJitter(0.1, 0.1, 0.1),
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
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD),
            ])),
        Extract(['image', 'meta', 'target']),
    ])

    return train_transform, eval_transform


def train_epoch(model, data_loader, optimizer, scheduler, epoch, config):
    metrics = {
        'images': Last(),
        'loss': Mean(),
        'lr': Last(),
    }

    # loop over batches ################################################################################################
    model.train()
    for images, meta, targets in \
            tqdm(data_loader, desc='fold {}, epoch {}/{}, train'.format(config.fold, epoch, config.train.epochs)):
        images, meta, targets = images.to(DEVICE), {k: meta[k].to(DEVICE) for k in meta}, targets.to(DEVICE)
        # images, targets = mix_up(images, targets, alpha=1.)

        logits = model(images, meta)
        loss = compute_loss(input=logits, target=targets, config=config)

        metrics['images'].update(images.data.cpu())
        metrics['loss'].update(loss.data.cpu())
        metrics['lr'].update(np.squeeze(scheduler.get_last_lr()))

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
        scheduler.step()

    # compute metrics ##################################################################################################
    with torch.no_grad():
        metrics = {k: metrics[k].compute_and_reset() for k in metrics}

        writer = SummaryWriter(os.path.join(config.experiment_path, 'train'))
        writer.add_image('images', torchvision.utils.make_grid(
            metrics['images'], nrow=compute_nrow(metrics['images']), normalize=True), global_step=epoch)
        writer.add_scalar('loss', metrics['loss'], global_step=epoch)
        writer.add_scalar('lr', metrics['lr'], global_step=epoch)

        writer.flush()
        writer.close()


def eval_epoch(model, data_loader, epoch, config, suffix=''):
    metrics = {
        'images': Concat(),
        'targets': Concat(),
        'logits': Concat(),
        'loss': Concat(),
    }

    # loop over batches ################################################################################################
    model.eval()
    with torch.no_grad():
        for images, meta, targets in \
                tqdm(data_loader, desc='fold {}, epoch {}/{}, eval'.format(config.fold, epoch, config.train.epochs)):
            images, meta, targets = images.to(DEVICE), {k: meta[k].to(DEVICE) for k in meta}, targets.to(DEVICE)

            logits = model(images, meta)
            loss = compute_loss(input=logits, target=targets, config=config)

            metrics['images'].update(images.data.cpu())
            metrics['targets'].update(targets.data.cpu())
            metrics['logits'].update(logits.data.cpu())
            metrics['loss'].update(loss.data.cpu())

    # compute metrics ##################################################################################################
    with torch.no_grad():
        metrics = {k: metrics[k].compute_and_reset() for k in metrics}
        metrics.update(compute_metric(input=metrics['logits'], target=metrics['targets']))
        images_hard_pos = topk_hardest(
            metrics['images'], metrics['loss'], metrics['targets'] > 0.5, topk=config.eval.batch_size)
        images_hard_neg = topk_hardest(
            metrics['images'], metrics['loss'], metrics['targets'] <= 0.5, topk=config.eval.batch_size)
        roc_curve = plot_roc_curve(input=metrics['logits'], target=metrics['targets'])
        metrics['loss'] = metrics['loss'].mean()

        writer = SummaryWriter(os.path.join(config.experiment_path, 'eval', suffix))
        writer.add_image('images/hard/pos', torchvision.utils.make_grid(
            images_hard_pos, nrow=compute_nrow(images_hard_pos), normalize=True), global_step=epoch)
        writer.add_image('images/hard/neg', torchvision.utils.make_grid(
            images_hard_neg, nrow=compute_nrow(images_hard_neg), normalize=True), global_step=epoch)
        writer.add_scalar('loss', metrics['loss'], global_step=epoch)
        writer.add_scalar('roc_auc', metrics['roc_auc'], global_step=epoch)
        writer.add_figure('roc_curve', roc_curve, global_step=epoch)

        writer.flush()
        writer.close()

    return metrics['roc_auc']


def compute_loss(input, target, config):
    def has_valid_size(x):
        return x.size() == (input.size(0),)

    def f(input, target, name):
        if name == 'ce':
            return sigmoid_cross_entropy(input=input, target=target)
        elif name == 'lsep':
            return lsep_loss(input=input, target=target)
        elif name == 'f1':
            return f1_loss(input=input.sigmoid(), target=target)
        else:
            raise AssertionError('invalid loss {}'.format(name))

    # loss = [
    #     # sigmoid_focal_loss(input=input, target=target),
    # ]

    loss = [
        f(input=input, target=target, name=name)
        for name in config.train.loss]
    assert all(map(has_valid_size, loss))
    loss = sum(loss)
    assert has_valid_size(loss)

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


def topk_hardest(input, loss, mask, topk):
    input, loss = input[mask], loss[mask]

    assert input.size(0) == loss.size(0)
    _, indices = torch.topk(loss, topk)

    return input[indices]


if __name__ == '__main__':
    main()
