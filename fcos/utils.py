from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import ImageFont, Image, ImageDraw

from object_detection.box_utils import boxes_clip
from utils import one_hot


class Detections(namedtuple('Detections', ['class_ids', 'boxes', 'scores'])):
    def to(self, device):
        return self.apply(lambda x: x.to(device))

    def apply(self, f):
        def apply(x):
            if x is None:
                return None
            else:
                return f(x)

        return Detections(
            class_ids=apply(self.class_ids),
            boxes=apply(self.boxes),
            scores=apply(self.scores))


def fill_scores(dets):
    assert dets.scores is None

    return Detections(
        class_ids=dets.class_ids,
        boxes=dets.boxes,
        scores=torch.ones_like(dets.class_ids, dtype=torch.float))


# TODO: fix boxes usage
def draw_boxes(image, detections, class_names, line_width=2, shade=True):
    font = ImageFont.truetype('./data/Droid+Sans+Mono+Awesome.ttf', size=14)

    detections = Detections(
        class_ids=detections.class_ids,
        boxes=boxes_clip(detections.boxes, image.size()[1:3]).round().long(),
        scores=detections.scores)

    device = image.device
    image = image.permute(1, 2, 0).data.cpu().numpy()

    if shade:
        mask = np.zeros_like(image, dtype=np.bool)
        for t, l, b, r in detections.boxes.data.cpu().numpy():
            mask[t:b, l:r] = True
        image = np.where(mask, image, image * 0.5)

    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)

    for c, (t, l, b, r), s in zip(
            detections.class_ids.data.cpu().numpy(),
            detections.boxes.data.cpu().numpy(),
            detections.scores.data.cpu().numpy()):
        if len(class_names) > 1:
            colors = np.random.RandomState(42).uniform(85, 255, size=(len(class_names), 3)).round().astype(np.uint8)
            color = tuple(colors[c])
            text = '{}: {:.2f}'.format(class_names[c], s)
            size = draw.textsize(text, font=font)
            draw.rectangle(((l, t - size[1]), (l + size[0] + line_width * 2, t)), fill=color)
            draw.text((l + line_width, t - size[1]), text, font=font, fill=(0, 0, 0))
        else:
            color = (s - 0.5) / 0.5
            color = color * np.array([255, 85, 85]) + (1 - color) * np.array([85, 85, 255])
            color = tuple(color.round().astype(np.uint8))
        draw.rectangle(((l, t), (r, b)), outline=color, width=line_width)

    image = np.array(image) / 255
    image = torch.tensor(image).permute(2, 0, 1).to(device)

    return image


def foreground_binary_coding(input, num_classes):
    return one_hot(input + 1, num_classes + 2)[..., 2:]


def pr_curve_plot(pr):
    fig = plt.figure()
    plt.plot(pr[:, 1], pr[:, 0])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.fill_between(pr[:, 1], 0, pr[:, 0], alpha=0.1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    return fig


def apply_recursively(f, x):
    if isinstance(x, Detections):
        return f(x)
    elif isinstance(x, list):
        return list(apply_recursively(f, y) for y in x)
    elif isinstance(x, tuple):
        return tuple(apply_recursively(f, y) for y in x)
    elif isinstance(x, dict):
        return {k: apply_recursively(f, x[k]) for k in x}
    else:
        return f(x)


def flatten_detection_map(input):
    *rest, c, h, w = input.size()
    assert 0 <= len(rest) <= 1
    input = input.view(*rest, c, h * w)
    input = input.transpose(-1, -2)

    return input


def replace_bn_with_gn(m):
    if isinstance(m, nn.BatchNorm2d):
        return nn.GroupNorm(num_channels=m.num_features, num_groups=32)

    for n, c in m.named_children():
        setattr(m, n, replace_bn_with_gn(c))

    return m
