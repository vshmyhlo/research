import os

import click
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms as T
from all_the_tools.config import load_config
from all_the_tools.metrics import Last, Mean
from all_the_tools.torch.utils import Saver
from sklearn.model_selection import StratifiedShuffleSplit
from tensorboardX import SummaryWriter
from tqdm import tqdm

from fix_match.utils import UDataset, XUDataLoader
from mix_match.model import Model
from utils import WarmupCosineAnnealingLR, compute_nrow, entropy, one_hot

NUM_CLASSES = 10
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MEAN = torch.tensor([0.4914, 0.4822, 0.4465])
STD = torch.tensor([0.2470, 0.2435, 0.2616])


@click.command()
@click.option('--config-path', type=click.Path(), required=True)
@click.option('--dataset-path', type=click.Path(), required=True)
@click.option('--experiment-path', type=click.Path(), required=True)
@click.option('--restore-path', type=click.Path())
@click.option('--workers', type=click.INT, default=os.cpu_count())
def main(config_path, **kwargs):
    config = load_config(
        config_path,
        **kwargs)

    # build transforms #################################################################################################
    transform_x, transform_u, eval_transform = build_transforms()

    # build datasets ###################################################################################################
    x_indices, u_indices = build_x_u_split(
        torchvision.datasets.CIFAR10(config.dataset_path, train=True, download=True),
        config.train.num_labeled)

    x_dataset = torch.utils.data.Subset(
        torchvision.datasets.CIFAR10(config.dataset_path, train=True, transform=transform_x, download=True),
        x_indices)
    u_dataset = UDataset(*[
        torch.utils.data.Subset(
            torchvision.datasets.CIFAR10(config.dataset_path, train=True, transform=transform_u, download=True),
            u_indices)
        for _ in range(2)
    ])
    eval_dataset = torchvision.datasets.CIFAR10(
        config.dataset_path, train=False, transform=eval_transform, download=True)

    # build data loaders ###############################################################################################
    train_data_loader = XUDataLoader(
        torch.utils.data.DataLoader(
            x_dataset,
            batch_size=config.train.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=config.workers),
        torch.utils.data.DataLoader(
            u_dataset,
            batch_size=config.train.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=config.workers))
    eval_data_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=config.eval.batch_size,
        num_workers=config.workers)

    # build model ######################################################################################################
    model = Model(config.model, NUM_CLASSES).to(DEVICE)
    model.apply(weights_init)
    optimizer = build_optimizer(model.parameters(), config)
    scheduler = build_scheduler(optimizer, config, len(train_data_loader))
    saver = Saver({
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
    })
    if config.restore_path is not None:
        saver.load(config.restore_path, keys=['model'])

    for epoch in range(1, config.epochs + 1):
        train_epoch(model, train_data_loader, optimizer, scheduler, epoch=epoch, config=config)
        if epoch % config.log_interval != 0:
            continue
        eval_epoch(model, eval_data_loader, epoch=epoch, config=config)
        saver.save(
            os.path.join(config.experiment_path, 'checkpoint_{}.pth'.format(epoch)),
            epoch=epoch)


def build_x_u_split(dataset, num_labeled):
    targets = torch.tensor([target for _, target in tqdm(dataset, 'loading split')])

    u_indices, x_indices = next(
        StratifiedShuffleSplit(n_splits=1, test_size=num_labeled, random_state=42).split(targets, targets))
    u_indices, x_indices = torch.tensor(u_indices), torch.tensor(x_indices)

    return x_indices, u_indices


def build_optimizer(parameters, config):
    if config.train.opt.type == 'sgd':
        optimizer = torch.optim.SGD(
            parameters,
            config.train.opt.lr,
            momentum=config.train.opt.momentum,
            weight_decay=config.train.opt.weight_decay,
            nesterov=True)
    elif config.train.opt.type == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            parameters,
            config.train.opt.lr,
            momentum=config.train.opt.momentum,
            weight_decay=config.train.opt.weight_decay)
    elif config.train.opt.type == 'adam':
        optimizer = torch.optim.Adam(
            parameters,
            config.train.opt.lr,
            weight_decay=config.train.opt.weight_decay)
    else:
        raise AssertionError('invalid optimizer {}'.format(config.train.opt.type))

    return optimizer


def build_scheduler(optimizer, config, steps_per_epoch):
    if config.train.sched.type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs * steps_per_epoch)
    elif config.train.sched.type == 'warmup_cosine':
        scheduler = WarmupCosineAnnealingLR(
            optimizer,
            epoch_warmup=config.epochs_warmup * steps_per_epoch,
            epoch_max=config.epochs * steps_per_epoch)
    elif config.train.sched.type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=(config.epochs * steps_per_epoch) // 3,
            gamma=0.1)
    elif config.train.sched.type == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            [epoch * steps_per_epoch for epoch in config.train.sched.epochs],
            gamma=0.1)
    else:
        raise AssertionError('invalid scheduler {}'.format(config.train.sched.type))

    return scheduler


def build_transforms():
    def validate_size(input):
        assert input.size() == (3, 32, 32)

        return input

    to_tensor_and_norm = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
        validate_size,
    ])
    transform_x = transform_u = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode='reflect'),
        to_tensor_and_norm,
    ])
    eval_transform = T.Compose([
        to_tensor_and_norm,
    ])

    return transform_x, transform_u, eval_transform


def compute_loss_x(input, target):
    assert input.size() == input.size()
    return -torch.sum(target * torch.log(input + 1e-8), 1)


def compute_loss_u(input, target):
    assert input.size() == input.size()
    return ((input - target)**2).mean(1)


def shuffle(images, targets):
    assert images.size(0) == targets.size(0)
    perm = np.random.permutation(images.size(0))

    return images[perm], targets[perm]


def sharpen(input, t):
    input = input**(1 / t)
    input /= input.sum(1, keepdim=True)

    return input


def mix_up(left, right, a):
    assert len(left) == len(right)
    assert all(l.size() == r.size() for l, r in zip(left, right))

    lam = torch.distributions.Beta(a, a).sample().to(left[0].device)
    lam = torch.max(lam, 1 - lam)

    return [lam * l + (1 - lam) * r for l, r in zip(left, right)]


def mix_match(x, u, model, config):
    images_x, targets_x = x
    images_u_0, images_u_1 = u
    del x, u

    # predict u targets ################################################################################################
    images_u = torch.cat([images_u_0, images_u_1], 0)
    targets_u_0, targets_u_1 = \
        model(images_u).detach() \
            .split([images_u_0.size(0), images_u_1.size(0)])

    targets_u = sharpen((targets_u_0 + targets_u_1) / 2, t=config.train.temp)
    targets_u = targets_u.repeat(2, 1)

    # shuffle ##########################################################################################################
    images_w = torch.cat([images_x, images_u], 0)
    targets_w = torch.cat([targets_x, targets_u], 0)
    images_w, targets_w = shuffle(images=images_w, targets=targets_w)
    images_w_0, images_w_1 = images_w.split([images_x.size(0), images_u.size(0)])
    targets_w_0, targets_w_1 = targets_w.split([targets_x.size(0), targets_u.size(0)])

    # mix-up ###########################################################################################################
    images_x, targets_x = mix_up(
        left=(images_x, targets_x),
        right=(images_w_0, targets_w_0),
        a=config.train.alpha)
    images_u, targets_u = mix_up(
        left=(images_u, targets_u),
        right=(images_w_1, targets_w_1),
        a=config.train.alpha)

    return (images_x, targets_x), (images_u, targets_u)


def train_epoch(model, data_loader, optimizer, scheduler, epoch, config):
    metrics = {
        'loss/x': Mean(),
        'loss/u': Mean(),
        'weight/u': Last(),
        'lr': Last(),
    }

    model.train()
    for (images_x, targets_x), (images_u_0, images_u_1) in \
            tqdm(data_loader, desc='epoch {}/{}, train'.format(epoch, config.epochs)):
        # prepare data #################################################################################################
        images_x, targets_x, images_u_0, images_u_1 = \
            images_x.to(DEVICE), targets_x.to(DEVICE), images_u_0.to(DEVICE), images_u_1.to(DEVICE)
        targets_x = one_hot(targets_x, NUM_CLASSES)

        with torch.no_grad():
            (images_x, targets_x), (images_u, targets_u) = \
                mix_match(
                    x=(images_x, targets_x),
                    u=(images_u_0, images_u_1),
                    model=model,
                    config=config)

        probs_x, probs_u = \
            model(torch.cat([images_x, images_u], 0)) \
                .split([images_x.size(0), images_u.size(0)])

        # x ############################################################################################################
        loss_x = compute_loss_x(input=probs_x, target=targets_x)
        metrics['loss/x'].update(loss_x.data.cpu().numpy())

        # u ############################################################################################################
        loss_u = compute_loss_u(input=probs_u, target=targets_u)
        metrics['loss/u'].update(loss_u.data.cpu().numpy())

        # opt step #####################################################################################################
        metrics['lr'].update(np.squeeze(scheduler.get_last_lr()))
        weight_u = config.train.weight_u * min((epoch - 1) / config.epochs_warmup, 1.)
        metrics['weight/u'].update(weight_u)

        optimizer.zero_grad()
        (loss_x.mean() + weight_u * loss_u.mean()).backward()
        optimizer.step()
        scheduler.step()

    if epoch % config.log_interval != 0:
        return

    writer = SummaryWriter(os.path.join(config.experiment_path, 'train'))
    with torch.no_grad():
        for k in metrics:
            writer.add_scalar(k, metrics[k].compute_and_reset(), global_step=epoch)
        writer.add_image('images_x', torchvision.utils.make_grid(
            images_x, nrow=compute_nrow(images_x), normalize=True), global_step=epoch)
        writer.add_image('images_u', torchvision.utils.make_grid(
            images_u, nrow=compute_nrow(images_u), normalize=True), global_step=epoch)

    writer.flush()
    writer.close()


def eval_epoch(model, data_loader, epoch, config):
    metrics = {
        'accuracy': Mean(),
        'entropy': Mean(),
    }

    with torch.no_grad():
        model.eval()
        for images_x, targets_x in tqdm(data_loader, desc='epoch {}/{}, eval'.format(epoch, config.epochs)):
            images_x, targets_x = images_x.to(DEVICE), targets_x.to(DEVICE)

            probs_x = model(images_x)

            metrics['entropy'].update(entropy(probs_x).data.cpu().numpy())
            metrics['accuracy'].update((probs_x.argmax(-1) == targets_x).float().data.cpu().numpy())

    writer = SummaryWriter(os.path.join(config.experiment_path, 'eval'))
    with torch.no_grad():
        for k in metrics:
            writer.add_scalar(k, metrics[k].compute_and_reset(), global_step=epoch)
        writer.add_image('images_x', torchvision.utils.make_grid(
            images_x, nrow=compute_nrow(images_x), normalize=True), global_step=epoch)

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


if __name__ == '__main__':
    main()
