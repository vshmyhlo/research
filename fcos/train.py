import argparse
import gc
import itertools
import os
import shutil

import numpy as np
import torch
import torch.distributions
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms as T
from all_the_tools.metrics import Mean, Last
from all_the_tools.torch.utils import Saver, seed_torch
from all_the_tools.transforms import ApplyTo
from all_the_tools.utils import seed_python
from tensorboardX import SummaryWriter
from tqdm import tqdm

from detection.anchor_utils import compute_anchor
from detection.box_coding import decode_boxes, shifts_scales_to_boxes, boxes_to_shifts_scales
from detection.box_utils import boxes_iou
from detection.config import build_default_config
from detection.datasets.coco import Dataset as CocoDataset
from detection.datasets.wider import Dataset as WiderDataset
from detection.losses import boxes_iou_loss, smooth_l1_loss, focal_loss
from detection.map import per_class_precision_recall_to_map
from detection.metrics import FPS, PerClassPR
from detection.model import RetinaNet
from detection.transform import Resize, BuildLabels, RandomCrop, RandomFlipLeftRight, denormalize, FilterBoxes
from detection.utils import draw_boxes, DataLoaderSlice, pr_curve_plot, fill_scores

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

parser = argparse.ArgumentParser()
parser.add_argument('--config-path', type=str, required=True)
parser.add_argument('--experiment-path', type=str, default='./tf_log/detection')
parser.add_argument('--dataset-path', type=str, required=True)
parser.add_argument('--restore-path', type=str)
parser.add_argument('--workers', type=int, default=os.cpu_count())
args = parser.parse_args()
config = build_default_config()
config.merge_from_file(args.config_path)
config.freeze()
os.makedirs(args.experiment_path, exist_ok=True)
shutil.copy(args.config_path, args.experiment_path)

if config.dataset == 'coco':
    Dataset = CocoDataset
elif config.dataset == 'wider':
    Dataset = WiderDataset
else:
    raise AssertionError('invalid config.dataset {}'.format(config.dataset))

ANCHOR_TYPES = list(itertools.product(config.anchors.ratios, config.anchors.scales))
ANCHORS = [
    [compute_anchor(size, ratio, scale) for ratio, scale in ANCHOR_TYPES]
    if size is not None else None
    for size in config.anchors.sizes
]


def worker_init_fn(_):
    seed_python(torch.initial_seed() % 2**32)


def compute_classification_loss(input, target):
    if input.numel() == 0:
        return torch.tensor(0.)

    if config.loss.classification == 'focal':
        loss = focal_loss(input=input, target=target)
    else:
        raise AssertionError('invalid config.loss.classification {}'.format(config.loss.classification))

    return loss.mean()


def compute_localization_loss(input, target, anchors):
    if input.numel() == 0:
        return torch.tensor(0.)

    if config.loss.localization == 'smooth_l1':
        target = boxes_to_shifts_scales(target, anchors)
        loss = smooth_l1_loss(input=input, target=target)
    elif config.loss.localization == 'iou':
        input = shifts_scales_to_boxes(input, anchors)
        loss = boxes_iou_loss(input=input, target=target)
    else:
        raise AssertionError('invalid config.loss.localization {}'.format(config.loss.localization))

    return loss.mean()


# TODO: check loss
# TODO: incostistent interface in compute_loss and compute_metric
# TODO: target_class, target_loc, target_boxes = target ?
def compute_loss(input, target, anchors):
    input_class, input_loc = input
    target_class, target_loc = target

    # classification loss
    class_mask = target_class != -1
    class_loss = compute_classification_loss(
        input=input_class[class_mask], target=target_class[class_mask])

    # localization loss
    loc_mask = target_class > 0
    loc_loss = compute_localization_loss(
        input=input_loc[loc_mask], target=target_loc[loc_mask], anchors=anchors[loc_mask])

    assert class_loss.size() == loc_loss.size()
    loss = class_loss + loc_loss

    return loss


def compute_metric(input, target):
    input_class, input_loc = input
    target_class, target_loc = target

    loc_mask = target_class > 0
    iou = boxes_iou(input_loc[loc_mask], target_loc[loc_mask])

    return {
        'iou': iou,
    }


def build_optimizer(parameters, config):
    if config.opt.type == 'sgd':
        return torch.optim.SGD(
            parameters,
            config.opt.learning_rate,
            momentum=config.opt.sgd.momentum,
            weight_decay=config.opt.weight_decay,
            nesterov=True)
    else:
        raise AssertionError('invalid config.opt.type {}'.format(config.opt.type))


def build_scheduler(optimizer, config, epoch_size, start_epoch):
    # FIXME:
    if config.sched.type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            config.epochs * epoch_size,
            last_epoch=start_epoch * epoch_size - 1)
    else:
        raise AssertionError('invalid config.sched.type {}'.format(config.sched.type))


def decode(output, anchors):
    output_class, output_loc = output
    output_loc = shifts_scales_to_boxes(output_loc, anchors)

    return output_class, output_loc


def train_epoch(model, optimizer, scheduler, data_loader, class_names, epoch):
    writer = SummaryWriter(os.path.join(args.experiment_path, 'train'))

    metrics = {
        'loss': Mean(),
        'learning_rate': Last(),
    }

    model.train()
    optimizer.zero_grad()
    for i, (images, labels, anchors, dets_true) in enumerate(tqdm(data_loader, desc='epoch {} train'.format(epoch)), 1):
        images, labels, anchors, dets_true = \
            images.to(DEVICE), [m.to(DEVICE) for m in labels], anchors.to(DEVICE), [d.to(DEVICE) for d in dets_true]
        output = model(images)

        loss = compute_loss(input=output, target=labels, anchors=anchors)
        metrics['loss'].update(loss.data.cpu().numpy())
        metrics['learning_rate'].update(np.squeeze(scheduler.get_lr()))

        output = decode(output, anchors)

        (loss.mean() / config.opt.acc_steps).backward()

        if i % config.opt.acc_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

    with torch.no_grad():
        metrics = {k: metrics[k].compute_and_reset() for k in metrics}
        print('[EPOCH {}][TRAIN] {}'.format(epoch, ', '.join('{}: {:.8f}'.format(k, metrics[k]) for k in metrics)))
        for k in metrics:
            writer.add_scalar(k, metrics[k], global_step=epoch)

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


def eval_epoch(model, data_loader, class_names, epoch):
    writer = SummaryWriter(os.path.join(args.experiment_path, 'eval'))

    metrics = {
        'loss': Mean(),
        'iou': Mean(),
        'fps': FPS(),
        'pr': PerClassPR(),
    }

    model.eval()
    with torch.no_grad():
        for images, labels, anchors, dets_true in tqdm(data_loader, desc='epoch {} evaluation'.format(epoch)):
            images, labels, anchors, dets_true = \
                images.to(DEVICE), [m.to(DEVICE) for m in labels], anchors.to(DEVICE), [d.to(DEVICE) for d in dets_true]
            output = model(images)

            loss = compute_loss(input=output, target=labels, anchors=anchors)
            metrics['loss'].update(loss.data.cpu().numpy())
            metrics['fps'].update(images.size(0))

            output = decode(output, anchors)

            dets_pred = [
                decode_boxes((c.sigmoid(), r))
                for c, r in zip(*output)]
            metrics['pr'].update((dets_true, dets_pred))

            metric = compute_metric(input=output, target=labels)
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
    images, labels, anchors, dets = zip(*batch)

    images = torch.utils.data.dataloader.default_collate(images)
    labels = torch.utils.data.dataloader.default_collate(labels)
    anchors = torch.utils.data.dataloader.default_collate(anchors)

    return images, labels, anchors, dets


def main():
    seed_python(config.seed)
    seed_torch(config.seed)

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
        BuildLabels(ANCHORS, min_iou=config.anchors.min_iou, max_iou=config.anchors.max_iou),
    ])
    eval_transform = T.Compose([
        Resize(config.resize_size),
        RandomCrop(config.crop_size),
        ApplyTo('image', T.Compose([
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ])),
        FilterBoxes(),
        BuildLabels(ANCHORS, min_iou=config.anchors.min_iou, max_iou=config.anchors.max_iou),
    ])

    train_dataset = Dataset(args.dataset_path, subset='train', transform=train_transform)
    eval_dataset = Dataset(args.dataset_path, subset='eval', transform=eval_transform)
    class_names = train_dataset.class_names

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn)
    if config.train_steps is not None:
        train_data_loader = DataLoaderSlice(train_data_loader, config.train_steps)
    eval_data_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=config.eval_batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn)

    model = RetinaNet(
        config.model,
        num_classes=Dataset.num_classes,
        anchors_per_level=len(ANCHOR_TYPES),
        anchor_levels=[a is not None for a in ANCHORS])
    model = model.to(DEVICE)

    optimizer = build_optimizer(model.parameters(), config)

    saver = Saver({'model': model, 'optimizer': optimizer})
    start_epoch = 0
    if args.restore_path is not None:
        saver.load(args.restore_path, keys=['model'])
    if os.path.exists(os.path.join(args.experiment_path, 'checkpoint.pth')):
        start_epoch = saver.load(os.path.join(args.experiment_path, 'checkpoint.pth'))

    scheduler = build_scheduler(optimizer, config, len(train_data_loader), start_epoch)

    for epoch in range(start_epoch, config.epochs):
        train_epoch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            data_loader=train_data_loader,
            class_names=class_names,
            epoch=epoch)
        gc.collect()
        eval_epoch(
            model=model,
            data_loader=eval_data_loader,
            class_names=class_names,
            epoch=epoch)
        gc.collect()

        saver.save(os.path.join(args.experiment_path, 'checkpoint.pth'), epoch=epoch + 1)


if __name__ == '__main__':
    main()
