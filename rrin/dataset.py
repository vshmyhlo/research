import os
import random

import pandas as pd
import torch.utils
import torchvision.transforms.functional as F
from PIL import Image


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        self.data = pd.read_csv(os.path.join(path, 'tri_trainlist.txt'), names=['folder'])
        self.data['folder'] = self.data['folder'].apply(lambda folder: os.path.join(path, 'sequences', folder))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        sample = self.data.iloc[i]['folder']
        sample = [os.path.join(sample, 'im{}.png'.format(n)) for n in [1, 2, 3]]
        sample = [Image.open(image) for image in sample]

        if self.transform is not None:
            sample = [self.transform(image) for image in sample]

        return sample


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
