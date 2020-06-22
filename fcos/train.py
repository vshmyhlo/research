import gc
import os

import click
import numpy as np
import torch
import torch.distributions
import torch.nn.functional as F
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms as T
from all_the_tools.config import load_config
from all_the_tools.metrics import Mean, Last
from all_the_tools.torch.utils import Saver
from all_the_tools.transforms import ApplyTo
from all_the_tools.utils import seed_python
from tensorboardX import SummaryWriter
from tqdm import tqdm

from fcos.box_coder import BoxCoder
from fcos.loss import compute_loss
# from detection.losses import boxes_iou_loss, smooth_l1_loss, focal_loss
# from detection.map import per_class_precision_recall_to_map
from fcos.metrics import FPS, PerClassPR
from fcos.model import FCOS
from fcos.transforms import Resize, BuildTargets, RandomCrop, RandomFlipLeftRight, denormalize, FilterBoxes
from fcos.utils import apply_recursively
# from detection.model import RetinaNet
from fcos.utils import draw_boxes
from fcos.utils import foreground_binary_coding
# from detection.anchor_utils import compute_anchor
# from detection.box_coding import decode_boxes, shifts_scales_to_boxes, boxes_to_shifts_scales
# from detection.box_utils import boxes_iou
# from detection.config import build_default_config
from object_detection.datasets.coco import Dataset as CocoDataset
from utils import random_seed, DataLoaderSlice
from utils import weighted_sum

# from detection.utils import draw_boxes, DataLoaderSlice, pr_curve_plot, fill_scores


# TODO: check encode-decode gives same result

# TODO: clip boxes in decoding?
# TODO: maybe use 1-based class indexing (maybe better not)
# TODO: check again order of anchors at each level
# TODO: eval on full-scale
# TODO: min/max object size filter
# TODO: boxfilter separate transform
# TODO: do not store c1 map
# TODO: compute metric with original boxes
# TODO: pin memory
# TODO: random resize
# TODO: plot box overlap distribution
# TODO: smaller model/larger image
# TODO: visualization scores sigmoid
# TODO: move logits slicing to helpers
# TODO: iou + l1
# TODO: freeze BN
# TODO: generate boxes from masks
# TODO: move scores decoding to loss
# TODO: use named tensors
# TODO: rename all usages of "maps"
# TODO: show preds before nms
# TODO: show pred heatmaps
# TODO: learn anchor sizes/scales
# TODO: filter response norm


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# parser = argparse.ArgumentParser()
# parser.add_argument('--config-path', type=str, required=True)
# parser.add_argument('--experiment-path', type=str, default='./tf_log/detection')
# parser.add_argument('--dataset-path', type=str, required=True)
# parser.add_argument('--restore-path', type=str)
# parser.add_argument('--workers', type=int, default=os.cpu_count())
# args = parser.parse_args()
# config = build_default_config()
# config.merge_from_file(args.config_path)
# config.freeze()
# os.makedirs(args.experiment_path, exist_ok=True)
# shutil.copy(args.config_path, args.experiment_path)
#
#
# ANCHOR_TYPES = list(itertools.product(config.anchors.ratios, config.anchors.scales))
# ANCHORS = [
#     [compute_anchor(size, ratio, scale) for ratio, scale in ANCHOR_TYPES]
#     if size is not None else None
#     for size in config.anchors.sizes
# ]


def worker_init_fn(_):
    seed_python(torch.initial_seed() % 2**32)


def compute_metric(input, target):
    input_class, input_loc = input
    target_class, target_loc = target

    loc_mask = target_class > 0
    iou = boxes_iou(input_loc[loc_mask], target_loc[loc_mask])

    return {
        'iou': iou,
    }


def build_optimizer(parameters, config):
    if config.train.opt.type == 'sgd':
        return torch.optim.SGD(
            parameters,
            config.train.opt.learning_rate,
            momentum=config.train.opt.momentum,
            weight_decay=config.train.opt.weight_decay,
            nesterov=True)
    else:
        raise AssertionError('invalid config.train.opt.type {}'.format(config.train.opt.type))


def build_scheduler(optimizer, config, epoch_size, start_epoch):
    # FIXME:
    if config.train.sched.type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.train.epochs * epoch_size,
            last_epoch=start_epoch * epoch_size - 1)
    elif config.train.sched.type == 'step':
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[e * epoch_size for e in config.train.sched.steps],
            gamma=0.1,
            last_epoch=start_epoch * epoch_size - 1)
    else:
        raise AssertionError('invalid config.train.sched.type {}'.format(config.train.sched.type))


def train_epoch(model, optimizer, scheduler, data_loader, box_coder, class_names, epoch, config):
    metrics = {
        'loss': Mean(),
        'learning_rate': Last(),
    }

    model.train()
    optimizer.zero_grad()
    for i, batch in enumerate(tqdm(data_loader, desc='epoch {} train'.format(epoch)), 1):
        images, targets, dets_true = apply_recursively(lambda x: x.to(DEVICE), batch)

        output = model(images)

        loss = compute_loss(input=output, target=targets)

        metrics['loss'].update(loss.data.cpu().numpy())
        metrics['learning_rate'].update(np.squeeze(scheduler.get_lr()))

        (loss.mean() / config.train.acc_steps).backward()
        if i % config.train.acc_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()

    with torch.no_grad():
        metrics = {k: metrics[k].compute_and_reset() for k in metrics}
        writer = SummaryWriter(os.path.join(config.experiment_path, 'train'))

        for k in metrics:
            writer.add_scalar(k, metrics[k], global_step=epoch)

        images = denormalize(images, mean=MEAN, std=STD)

        dets_true = [
            box_coder.decode(foreground_binary_coding(c, 80), r, images.size()[2:])
            for c, r, s in zip(*targets)]
        dets_pred = [
            box_coder.decode(c.sigmoid(), r, images.size()[2:])
            for c, r, s in zip(*output)]

        true = [draw_boxes(i, d, class_names) for i, d in zip(images, dets_true)]
        pred = [draw_boxes(i, d, class_names) for i, d in zip(images, dets_pred)]

        writer.add_image(
            'detections/true',
            torchvision.utils.make_grid(true, nrow=4),
            global_step=epoch)
        writer.add_image(
            'detections/pred',
            torchvision.utils.make_grid(pred, nrow=4),
            global_step=epoch)

        writer.flush()
        writer.close()


def eval_epoch(model, data_loader, class_names, epoch, config):
    writer = SummaryWriter(os.path.join(config.experiment_path, 'eval'))

    metrics = {
        'loss': Mean(),
        'iou': Mean(),
        'fps': FPS(),
        'pr': PerClassPR(),
    }

    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='epoch {} evaluation'.format(epoch)):
            images, targets, dets_true = apply_recursively(lambda x: x.to(DEVICE), batch)

            output = model(images)

            loss = compute_loss(input=output, target=targets)

            metrics['loss'].update(loss.data.cpu().numpy())
            metrics['fps'].update(images.size(0))

            output = decode(output, anchors)

            dets_pred = [
                decode_boxes((c.sigmoid(), r))
                for c, r in zip(*output)]
            metrics['pr'].update((dets_true, dets_pred))

            metric = compute_metric(input=output, target=targets)
            for k in metric:
                metrics[k].update(metric[k].data.cpu().numpy())

        metrics = {k: metrics[k].compute_and_reset() for k in metrics}
        pr = metrics['pr']
        del metrics['pr']
        metrics['map'] = per_class_precision_recall_to_map(pr)
        print('[EPOCH {}][EVAL] {}'.format(epoch, ', '.join('{}: {:.8f}'.format(k, metrics[k]) for k in metrics)))
        for k in metrics:
            writer.add_scalar(k, metrics[k], global_step=epoch)

        for class_id in pr:
            writer.add_figure(
                'pr/{}'.format(class_names[class_id]), pr_curve_plot(pr[class_id]), global_step=epoch)

        dets_pred = [
            decode_boxes((c.sigmoid(), r))
            for c, r in zip(*output)]
        images_true = [
            draw_boxes(denormalize(i, mean=MEAN, std=STD), fill_scores(d), class_names)
            for i, d in zip(images, dets_true)]
        images_pred = [
            draw_boxes(denormalize(i, mean=MEAN, std=STD), d, class_names)
            for i, d in zip(images, dets_pred)]

        writer.add_image(
            'images_true', torchvision.utils.make_grid(images_true, nrow=4, normalize=True), global_step=epoch)
        writer.add_image(
            'images_pred', torchvision.utils.make_grid(images_pred, nrow=4, normalize=True), global_step=epoch)


def collate_fn(batch):
    images, targets, dets = zip(*batch)

    images = torch.utils.data.dataloader.default_collate(images)
    targets = torch.utils.data.dataloader.default_collate(targets)

    return images, targets, dets


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
    del kwargs
    random_seed(config.seed)

    box_coder = BoxCoder(config.model.levels)

    train_transform = T.Compose([
        Resize(config.resize_size),
        RandomCrop(config.crop_size),
        RandomFlipLeftRight(),
        ApplyTo('image', T.Compose([
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ])),
        FilterBoxes(),
        BuildTargets(box_coder),
    ])
    eval_transform = T.Compose([
        Resize(config.resize_size),
        RandomCrop(config.crop_size),
        ApplyTo('image', T.Compose([
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ])),
        FilterBoxes(),
        BuildTargets(box_coder),
    ])

    if config.dataset == 'coco':
        Dataset = CocoDataset
    else:
        raise AssertionError('invalid config.dataset {}'.format(config.dataset))
    train_dataset = Dataset(config.dataset_path, subset='train', transform=train_transform)
    eval_dataset = Dataset(config.dataset_path, subset='eval', transform=eval_transform)
    class_names = train_dataset.class_names

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=config.workers,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn)
    if config.train_steps is not None:
        train_data_loader = DataLoaderSlice(train_data_loader, config.train_steps)
    eval_data_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=config.eval.batch_size,
        drop_last=False,
        shuffle=True,
        num_workers=config.workers,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn)

    model = FCOS(config.model, num_classes=Dataset.num_classes)
    model = model.to(DEVICE)

    optimizer = build_optimizer(model.parameters(), config)

    saver = Saver({'model': model, 'optimizer': optimizer})
    start_epoch = 0
    if config.restore_path is not None:
        saver.load(config.restore_path, keys=['model'])
    if os.path.exists(os.path.join(config.experiment_path, 'checkpoint.pth')):
        start_epoch = saver.load(os.path.join(config.experiment_path, 'checkpoint.pth'))

    scheduler = build_scheduler(optimizer, config, len(train_data_loader), start_epoch)

    for epoch in range(start_epoch, config.train.epochs):
        train_epoch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            data_loader=train_data_loader,
            box_coder=box_coder,
            class_names=class_names,
            epoch=epoch,
            config=config)
        gc.collect()
        # eval_epoch(
        #     model=model,
        #     data_loader=eval_data_loader,
        #     box_coder=box_coder,
        #     class_names=class_names,
        #     epoch=epoch,
        #     config=config)
        gc.collect()

        saver.save(os.path.join(config.experiment_path, 'checkpoint.pth'), epoch=epoch + 1)


def draw_class_map(image, class_map, num_classes):
    colors = np.random.RandomState(42).uniform(1 / 3, 1, size=(num_classes + 1, 3))
    colors[0] = 0.
    colors = torch.tensor(colors, dtype=torch.float, device=class_map.device)

    class_map = colors[class_map]
    class_map = class_map.permute(0, 3, 1, 2)
    class_map = F.interpolate(class_map, size=image.size()[2:], mode='nearest')

    return weighted_sum(image, class_map, 0.5)


if __name__ == '__main__':
    main()
