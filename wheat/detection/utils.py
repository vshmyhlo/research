from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import ImageFont, Image, ImageDraw
from all_the_tools.torch.utils import one_hot

from detection.box_utils import boxes_clip


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


class DataLoaderSlice(object):
    def __init__(self, data_loader, size):
        self.data_loader = data_loader
        self.size = size
        self.iter = None

    def __len__(self):
        return self.size

    def __iter__(self):
        i = 0
        while i < self.size:
            if self.iter is None:
                self.iter = iter(self.data_loader)

            try:
                yield next(self.iter)
                i += 1
            except StopIteration:
                self.iter = None


def logit(input):
    return torch.log(input / (1 - input))


def fill_scores(detections):
    assert detections.scores is None

    return Detections(
        class_ids=detections.class_ids,
        boxes=detections.boxes,
        scores=torch.ones_like(detections.class_ids, dtype=torch.float))


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
    return one_hot(input + 1, num_classes + 2)[:, 2:]


def pr_curve_plot(pr):
    fig = plt.figure()
    plt.plot(pr[:, 1], pr[:, 0])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.fill_between(pr[:, 1], 0, pr[:, 0], alpha=0.1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    return fig
