import random

import torchvision.transforms.functional as F
from PIL import Image


class RandomTranspose(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, input):
        if random.random() < self.p:
            return transpose(input)

        return input


def transpose(image):
    if not F._is_pil_image(image):
        raise TypeError('image should be PIL Image. Got {}'.format(type(image)))

    return image.transpose(Image.TRANSPOSE)
