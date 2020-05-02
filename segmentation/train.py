import importlib.util
import math
import os

import click
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision
import torchvision.transforms as T
from all_the_tools.metrics import Last, Mean, Metric
from all_the_tools.torch.losses import dice_loss, softmax_cross_entropy
from all_the_tools.torch.optim import LookAhead
from all_the_tools.torch.utils import Saver
from all_the_tools.transforms import Extract, ApplyTo
from all_the_tools.utils import seed_python
from tensorboardX import SummaryWriter
from tqdm import tqdm

from dataset import load_labeled_data, Dataset
from model import UNet as Model
from transforms import ToTensor, RandomHorizontalFlip

# TODO: maybe clip bad losses
# TODO: ewa of loss/wights and use it to reweight
# TODO: seresnext
# TODO: predict full character
# TODO: segment and classify
# TODO: synthetic data
# TODO: reweight loss based on how bad sample is
# TODO: cutout
# TODO: filter-response normalization
# TODO: train multiple models and make noisy student/teacher
# TODO: ewa optimization
# TODO: tf_efficientnet_b1_ns
# TODO: random crops
# TODO: progressive crops
# TODO: remove bad samples
# TODO: crop sides
# TODO: dropout
# TODO: pseudo label
# TODO: mixmatch
# TODO: do not use OHEM samples
# TODO: median ohem
# TODO: online hard examples mininig
# TODO: make label smoothing scale with loss
# TODO: label smoothing
# TODO: focal, weighting, lsep
# TODO: no soft recall
# TODO: self-mixup, self-cutmix
# TODO: population based training
# TODO: mixmatch
# TODO: stn regularize to identity
# TODO: pl
# TODO: spatial transformer network !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# TODO: dropout
# TODO: postprocess coocurence
# TODO: manifold mixup
# TODO: no soft f1 when trained with self-dist or LS
# TODO: nonlinear heads for classes
# TODO: lsep loss
# TODO: cutmix: progressive
# TODO: distillation
# TODO: erosion/dilation
# TODO: perspective
# TODO: shear/scale and other affine


# TODO: skeletonize and add noize
# TODO: scale
# TODO: mixmatch
# TODO: shift
# TODO: per-component metric
# TODO: center images
# TODO: synthetic data
# TODO: ohem
# TODO: ewa over distillation targets
# TODO: ewa over distillation mixing coefficient

NUM_CLASSES = 2
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class IoU(Metric):
    def compute(self):
        print(self.intersection.shape, self.union.shape)
        return (self.intersection / self.union).mean()

    def update(self, input, target):
        input = torch.eye(NUM_CLASSES, dtype=torch.bool, device=input.device)[input]
        target = torch.eye(NUM_CLASSES, dtype=torch.bool, device=target.device)[target]

        self.intersection += (input & target).sum((0, 1, 2)).data.cpu().numpy()
        self.union += (input | target).sum((0, 1, 2)).data.cpu().numpy()

    def reset(self):
        self.intersection = 0
        self.union = 0


def compute_weight_from_loss(loss):
    weight = 1 - 1 / (1 + loss)
    weight = weight / weight.mean()

    return weight


def compute_nrow(images):
    b, _, h, w = images.size()
    nrow = math.ceil(math.sqrt(h * b / w))

    return nrow


def weighted_sum(a, b, w):
    return w * a + (1 - w) * b


def worker_init_fn(_):
    seed_python(torch.initial_seed() % 2**32)


def softmax_recall_loss(input, target, dim=-1, eps=1e-7):
    assert input.dim() == 2
    assert input.size() == target.size()

    input = input.softmax(dim)

    tp = (target * input).sum()
    fn = (target * (1 - input)).sum()
    r = tp / (tp + fn + eps)

    loss = 1 - r

    return loss


def compute_loss(input, target, config):
    ce = softmax_cross_entropy(input, target, axis=1)
    ce = ce.mean((0, 1, 2))

    dice = dice_loss(input.softmax(1), target, smooth=0, axis=(0, 2, 3))
    dice = dice.mean(0)

    loss = ce + dice

    return loss


def train_eval_split(data, seed, eval_size=5000):
    indices = np.random.RandomState(seed).permutation(len(data))
    train_indices = indices[eval_size:]  # [:100]
    eval_indices = indices[:eval_size]
    print('split: {}, {}'.format(len(train_indices), len(eval_indices)))

    return train_indices, eval_indices


def build_optimizer(parameters, config):
    if config.type == 'sgd':
        optimizer = torch.optim.SGD(
            parameters,
            config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=True)
    elif config.type == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            parameters,
            config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay)
    elif config.type == 'adam':
        optimizer = torch.optim.Adam(
            parameters,
            config.lr,
            weight_decay=config.weight_decay)
    else:
        raise AssertionError('invalid optimizer {}'.format(config.type))

    if config.lookahead is not None:
        optimizer = LookAhead(
            optimizer,
            lr=config.lookahead.lr,
            num_steps=config.lookahead.steps)

    return optimizer


def build_scheduler(optimizer, config, optimizer_config, epochs, steps_per_epoch):
    if config.type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * steps_per_epoch)
    elif config.type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=(epochs * steps_per_epoch) // 3,
            gamma=0.1)
    else:
        raise AssertionError('invalid scheduler {}'.format(config.type))

    return scheduler


def load_config(config_path, **kwargs):
    spec = importlib.util.spec_from_file_location('config', config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.config
    for k in kwargs:
        setattr(config, k, kwargs[k])

    return config


def draw_masks(input):
    colors = np.random.RandomState(42).uniform(0.25, 1., size=(NUM_CLASSES, 3))
    colors = torch.tensor(colors, dtype=torch.float).to(input.device)
    colors[0, :] = 0.
    if NUM_CLASSES == 2:
        colors[1, :] = 1.

    input = colors[input]
    input = input.squeeze(1).permute(0, 3, 1, 2)

    return input


def denormalize(input):
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=input.dtype, device=input.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=input.dtype, device=input.device).view(1, 3, 1, 1)
    input = input * std + mean

    return input


def build_transforms():
    pre_processing = T.Compose([
        ApplyTo('image', T.Resize((1024 // 4, 1920 // 4))),
        ApplyTo('mask', T.Resize((1024 // 4, 1920 // 4))),
    ])
    post_processing = T.Compose([
        ToTensor(NUM_CLASSES),
        ApplyTo(
            'image',
            T.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225))),
        Extract(['image', 'mask']),
    ])
  
    train_transform = T.Compose([
        pre_processing,
        ApplyTo('image', T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)),
        RandomHorizontalFlip(),
        post_processing,
    ])
    eval_transform = T.Compose([
        pre_processing,
        post_processing,
    ])

    return train_transform, eval_transform


def train_epoch(model, data_loader, optimizer, scheduler, epoch, config):
    writer = SummaryWriter(os.path.join(config.experiment_path, 'train'))
    metrics = {
        'loss': Mean(),
        'lr': Last(),
    }

    model.train()
    for images, targets in tqdm(data_loader, desc='[epoch {}/{}] train'.format(epoch, config.epochs)):
        images, targets = images.to(DEVICE), targets.to(DEVICE)

        logits = model(images)
        loss = compute_loss(input=logits, target=targets, config=config.train)

        metrics['loss'].update(loss.data.cpu().numpy())
        metrics['lr'].update(np.squeeze(scheduler.get_lr()))

        loss.mean().backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    with torch.no_grad():
        mask_true = F.interpolate(draw_masks(targets.argmax(1, keepdim=True)), scale_factor=1)
        mask_pred = F.interpolate(draw_masks(logits.argmax(1, keepdim=True)), scale_factor=1)

        for k in metrics:
            writer.add_scalar(k, metrics[k].compute_and_reset(), global_step=epoch)
        writer.add_image('images', torchvision.utils.make_grid(
            denormalize(images), nrow=compute_nrow(images), normalize=True), global_step=epoch)
        # writer.add_image('mask_true', torchvision.utils.make_grid(
        #     mask_true, nrow=compute_nrow(mask_true), normalize=True), global_step=epoch)
        # writer.add_image('mask_pred', torchvision.utils.make_grid(
        #     mask_pred, nrow=compute_nrow(mask_pred), normalize=True), global_step=epoch)
        writer.add_image('images_true', torchvision.utils.make_grid(
            denormalize(images) + mask_true, nrow=compute_nrow(images), normalize=True), global_step=epoch)
        writer.add_image('images_pred', torchvision.utils.make_grid(
            denormalize(images) + mask_pred, nrow=compute_nrow(images), normalize=True), global_step=epoch)

    writer.flush()
    writer.close()


def eval_epoch(model, data_loader, epoch, config):
    writer = SummaryWriter(os.path.join(config.experiment_path, 'eval'))
    metrics = {
        'loss': Mean(),
        'iou': IoU(),
    }

    with torch.no_grad():
        model.eval()
        for images, targets in tqdm(data_loader, desc='[epoch {}/{}] eval'.format(epoch, config.epochs)):
            images, targets = images.to(DEVICE), targets.to(DEVICE)

            logits = model(images)
            loss = compute_loss(input=logits, target=targets, config=config.train)

            metrics['loss'].update(loss.data.cpu().numpy())
            metrics['iou'].update(
                input=logits.argmax(1),
                target=targets.argmax(1))

    with torch.no_grad():
        mask_true = F.interpolate(draw_masks(targets.argmax(1, keepdim=True)), scale_factor=1)
        mask_pred = F.interpolate(draw_masks(logits.argmax(1, keepdim=True)), scale_factor=1)

        for k in metrics:
            writer.add_scalar(k, metrics[k].compute_and_reset(), global_step=epoch)
        writer.add_image('images', torchvision.utils.make_grid(
            denormalize(images), nrow=compute_nrow(images), normalize=True), global_step=epoch)
        # writer.add_image('mask_true', torchvision.utils.make_grid(
        #     mask_true, nrow=compute_nrow(mask_true), normalize=True), global_step=epoch)
        # writer.add_image('mask_pred', torchvision.utils.make_grid(
        #     mask_pred, nrow=compute_nrow(mask_pred), normalize=True), global_step=epoch)
        writer.add_image('images_true', torchvision.utils.make_grid(
            denormalize(images) + mask_true, nrow=compute_nrow(images), normalize=True), global_step=epoch)
        writer.add_image('images_pred', torchvision.utils.make_grid(
            denormalize(images) + mask_pred, nrow=compute_nrow(images), normalize=True), global_step=epoch)

    writer.flush()
    writer.close()


@click.command()
@click.option('--config-path', type=click.Path(), required=True)
@click.option('--dataset-path', type=click.Tuple([click.Path(), click.Path()]), required=True)
@click.option('--experiment-path', type=click.Path(), required=True)
@click.option('--restore-path', type=click.Path())
@click.option('--lr-search', is_flag=True)
@click.option('--workers', type=click.INT, default=os.cpu_count())
def main(**kwargs):
    # TODO: seed everything
    config = load_config(**kwargs)  # FIXME:
    del kwargs

    train_eval_data = load_labeled_data(*config.dataset_path)

    train_indices, eval_indices = train_eval_split(train_eval_data, seed=config.seed)

    train_transform, eval_transform = build_transforms()

    train_dataset = Dataset(train_eval_data.iloc[train_indices], transform=train_transform)
    eval_dataset = Dataset(train_eval_data.iloc[eval_indices], transform=eval_transform)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=config.workers,
        worker_init_fn=worker_init_fn)
    eval_data_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=config.eval.batch_size,
        num_workers=config.workers,
        worker_init_fn=worker_init_fn)

    model = Model(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = build_optimizer(
        model.parameters(), config.train.optimizer)
    scheduler = build_scheduler(
        optimizer, config.train.scheduler, config.train.optimizer, config.epochs, len(train_data_loader))
    saver = Saver({
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
    })
    if config.restore_path is not None:
        saver.load(config.restore_path, keys=['model'])

    for epoch in range(1, config.epochs + 1):
        train_epoch(model, train_data_loader, optimizer, scheduler, epoch=epoch, config=config)
        eval_epoch(model, eval_data_loader, epoch=epoch, config=config)
        saver.save(
            os.path.join(
                config.experiment_path,
                'checkpoint_{}.pth'.format(epoch)),
            epoch=epoch)


if __name__ == '__main__':
    main()
