import importlib.util
import os

import click
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision
import torchvision.transforms as T
from all_the_tools.metrics import Last, Mean, Metric
from all_the_tools.torch.losses import softmax_cross_entropy
from all_the_tools.torch.utils import Saver
from all_the_tools.transforms import Extract, ApplyTo
from all_the_tools.utils import seed_python
from tensorboardX import SummaryWriter
from tqdm import tqdm

from segmentation.dataset import ADE20KDataset
from segmentation.model import UNet as Model
from segmentation.transforms import ToTensor, RandomHorizontalFlip, Resize, RandomCrop
from utils import compute_nrow

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

NUM_CLASSES = 150 + 1
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

    train_transform, eval_transform = build_transforms()

    train_data_loader = torch.utils.data.DataLoader(
        ADE20KDataset(config.dataset_path, subset='training', transform=train_transform),
        batch_size=config.train.batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=config.workers,
        worker_init_fn=worker_init_fn)
    eval_data_loader = torch.utils.data.DataLoader(
        ADE20KDataset(config.dataset_path, subset='validation', transform=eval_transform),
        batch_size=config.eval.batch_size,
        num_workers=config.workers,
        worker_init_fn=worker_init_fn)

    model = Model(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = build_optimizer(model.parameters(), config.train.optimizer)
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
            os.path.join(config.experiment_path, 'checkpoint_{}.pth'.format(epoch)),
            epoch=epoch)


def worker_init_fn(_):
    seed_python(torch.initial_seed() % 2**32)


def compute_loss(input, target):
    ce = softmax_cross_entropy(input, target, dim=1)
    ce = ce.mean((0, 1, 2))

    loss = ce

    return loss


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


def image_one_hot(input, n, dtype=torch.float):
    return torch.eye(n, dtype=dtype, device=input.device)[input].permute(0, 3, 1, 2)


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
    pre_process = Resize(256)
    post_process = T.Compose([
        ToTensor(NUM_CLASSES),
        ApplyTo(
            'image',
            T.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225))),
        Extract(['image', 'mask']),
    ])
    train_transform = T.Compose([
        pre_process,
        RandomCrop(256),
        RandomHorizontalFlip(),
        # ApplyTo('image', T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)),
        post_process,
    ])
    eval_transform = T.Compose([
        pre_process,
        post_process,
    ])

    return train_transform, eval_transform


def train_epoch(model, data_loader, optimizer, scheduler, epoch, config):
    writer = SummaryWriter(os.path.join(config.experiment_path, 'train'))
    metrics = {
        'loss': Mean(),
        'lr': Last(),
    }

    model.train()
    for images, targets in tqdm(data_loader, desc='epoch {}/{}, train'.format(epoch, config.epochs)):
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        targets = image_one_hot(targets, NUM_CLASSES)

        logits = model(images)
        loss = compute_loss(input=logits, target=targets)

        metrics['loss'].update(loss.data.cpu().numpy())
        metrics['lr'].update(np.squeeze(scheduler.get_lr()))

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
        scheduler.step()

    with torch.no_grad():
        mask_true = F.interpolate(draw_masks(targets.argmax(1, keepdim=True)), scale_factor=1)
        mask_pred = F.interpolate(draw_masks(logits.argmax(1, keepdim=True)), scale_factor=1)

        for k in metrics:
            writer.add_scalar(k, metrics[k].compute_and_reset(), global_step=epoch)
        writer.add_image('images', torchvision.utils.make_grid(
            denormalize(images), nrow=compute_nrow(images), normalize=True), global_step=epoch)
        writer.add_image('mask_true', torchvision.utils.make_grid(
            mask_true, nrow=compute_nrow(mask_true), normalize=True), global_step=epoch)
        writer.add_image('mask_pred', torchvision.utils.make_grid(
            mask_pred, nrow=compute_nrow(mask_pred), normalize=True), global_step=epoch)
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
        for images, targets in tqdm(data_loader, desc='epoch {}/{}, eval'.format(epoch, config.epochs)):
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            targets = image_one_hot(targets, NUM_CLASSES)

            logits = model(images)
            loss = compute_loss(input=logits, target=targets)

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
        writer.add_image('mask_true', torchvision.utils.make_grid(
            mask_true, nrow=compute_nrow(mask_true), normalize=True), global_step=epoch)
        writer.add_image('mask_pred', torchvision.utils.make_grid(
            mask_pred, nrow=compute_nrow(mask_pred), normalize=True), global_step=epoch)
        writer.add_image('images_true', torchvision.utils.make_grid(
            denormalize(images) + mask_true, nrow=compute_nrow(images), normalize=True), global_step=epoch)
        writer.add_image('images_pred', torchvision.utils.make_grid(
            denormalize(images) + mask_pred, nrow=compute_nrow(images), normalize=True), global_step=epoch)

    writer.flush()
    writer.close()


if __name__ == '__main__':
    main()
