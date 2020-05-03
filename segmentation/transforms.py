import numbers
import random
from typing import Iterable

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.transforms.transforms import _get_image_size


class ToTensor(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, input):
        image = F.to_tensor(input['image'])

        mask = np.array(input['mask'], dtype=np.int64)
        mask = torch.from_numpy(mask)

        input = {
            **input,
            'image': image,
            'mask': mask,
        }

        return input


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, input):
        if random.random() < self.p:
            input = {
                **input,
                'image': F.hflip(input['image']),
                'mask': F.hflip(input['mask'])
            }

        return input


class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, input):
        input = {
            **input,
            'image': F.resize(input['image'], self.size, self.interpolation),
            'mask': F.resize(input['mask'], self.size, Image.NEAREST),
        }

        return input


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, input):
        input = {
            **input,
            'image': F.center_crop(input['image'], self.size),
            'mask': F.center_crop(input['mask'], self.size),
        }

        return input


class RandomCrop(object):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, input):
        params = self.get_params(input['image'], self.size)

        input = {
            **input,
            'image': self.apply(input['image'], params),
            'mask': self.apply(input['mask'], params),
        }

        return input

    @staticmethod
    def get_params(img, output_size):
        w, h = _get_image_size(img)
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def apply(self, image, params):
        if self.padding is not None:
            image = F.pad(image, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and image.size[0] < self.size[1]:
            image = F.pad(image, (self.size[1] - image.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and image.size[1] < self.size[0]:
            image = F.pad(image, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)

        i, j, h, w = params

        return F.crop(image, i, j, h, w)
