import os

import cv2
import numpy as np
import torchvision.transforms as T
from PIL import Image
from PIL.ImageDraw import Draw


class LoadImage(object):
    def __init__(self, transform):
        self.transform = transform
        self.cache_path = './cache-mela-images/{}'.format(self.transform)

        os.makedirs(self.cache_path, exist_ok=True)

    def __call__(self, input):
        cache_path = os.path.join(self.cache_path, '{}.png'.format(input['id']))

        if not os.path.exists(cache_path):
            # dicom = pydicom.dcmread(input['image'])
            # image = Image.fromarray(dicom.pixel_array)
            image = Image.open(input['image'])
            image = self.transform(image)
            image.save(cache_path)
            del image

        image = Image.open(cache_path)

        input = {
            **input,
            'image': image,
        }

        return input


class ColorConstancy(object):
    def __call__(self, input):
        input = np.array(input)
        input = color_constancy(input)
        input = Image.fromarray(input)

        return input


class RandomResizedCrop(object):
    def __init__(self, size, scale=(0.875, 1 / 0.875)):
        self.size = size
        self.scale = scale

    def __call__(self, input):
        p = np.random.uniform(np.log2(self.scale[0]), np.log2(self.scale[1]))

        crop_size = round(self.size * 2**p)

        input = T.RandomCrop(crop_size)(input)
        input = T.Resize(self.size)(input)

        return input


def color_constancy(input, power=6, gamma=None):
    input = cv2.cvtColor(input, cv2.COLOR_RGB2BGR)
    dtype = input.dtype

    if gamma is not None:
        input = input.astype(np.uint8)
        look_up_table = np.ones((256, 1), dtype=np.uint8) * 0
        for i in range(256):
            look_up_table[i][0] = 255 * pow(i / 255, 1 / gamma)
        input = cv2.LUT(input, look_up_table)

    input = input.astype(np.float32)
    image_power = np.power(input, power)
    rgb_vec = np.power(np.mean(image_power, (0, 1)), 1 / power)
    rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
    rgb_vec = rgb_vec / rgb_norm
    rgb_vec = 1 / (rgb_vec * np.sqrt(3))
    input = np.multiply(input, rgb_vec)

    input = input.astype(dtype)
    input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)

    return input


class CircleMask(object):
    def __init__(self, color):
        self.color = tuple(color)

    def __call__(self, input):
        diam = min(input.size)
        lt = tuple((s - diam) // 2 for s in input.size)
        rb = tuple(x + diam for x in lt)

        mask = Image.new('1', input.size)
        Draw(mask).ellipse((lt, rb), fill=True)
        fill = Image.new(input.mode, input.size, color=self.color)
        input = Image.composite(input, fill, mask)

        return input
