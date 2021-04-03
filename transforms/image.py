import random

import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image


class RandomTranspose(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, input):
        if random.random() < self.p:
            return transpose(input)

        return input


class RandomRotate4(object):
    def __call__(self, input):
        angle = random.choice([90 * i for i in range(4)])

        return F.rotate(input, angle)


class Random8(T.Compose):
    def __init__(self):
        super().__init__(
            [
                RandomRotate4(),
                T.RandomVerticalFlip(),
            ]
        )


class TTA8(object):
    def __call__(self, input):
        input = [F.rotate(input, 90 * i) for i in range(4)]

        input = input + [F.vflip(x) for x in input]

        return input


def transpose(image):
    if not F._is_pil_image(image):
        raise TypeError("image should be PIL Image. Got {}".format(type(image)))

    return image.transpose(Image.TRANSPOSE)
