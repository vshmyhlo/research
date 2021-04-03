import os
import random

import pandas as pd
import torch.utils
import torchvision.transforms.functional as F
from PIL import Image


class Vimeo90kDataset(torch.utils.data.Dataset):
    def __init__(self, path, subset, transform=None):
        self.data = pd.read_csv(
            os.path.join(path, "tri_{}list.txt".format(subset)), names=["folder"]
        )
        self.data["folder"] = self.data["folder"].apply(
            lambda folder: os.path.join(path, "sequences", folder)
        )
        self.data.index = range(len(self.data))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        images = self.data.loc[i]["folder"]
        images = [os.path.join(images, "im{}.png".format(n)) for n in [1, 2, 3]]
        images = [Image.open(image) for image in images]

        if self.transform is not None:
            images = self.transform(images)

        return images


class MiddleburyDataset(torch.utils.data.Dataset):
    def __init__(self, path, video, transform=None):
        self.data = sorted(os.listdir(os.path.join(path, video)))
        self.data = [os.path.join(path, video, image) for image in self.data]
        self.data = pd.DataFrame(
            {
                "image_1": self.data[:-1],
                "image_3": self.data[1:],
            }
        )
        self.data.index = range(len(self.data))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        images = self.data.loc[i]
        images = [images["image_{}".format(i)] for i in [1, 3]]
        images = [Image.open(image) for image in images]

        if self.transform is not None:
            images = self.transform(images)

        return images


class ForEach(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, images):
        return [self.transform(image) for image in images]


class RandomTemporalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, images):
        if random.random() < self.p:
            images = reversed(images)

        return images


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, images):
        if random.random() < self.p:
            images = [F.hflip(image) for image in images]

        return images
