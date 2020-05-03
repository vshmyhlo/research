import glob
import os

import pandas as pd
import torch.utils.data
from PIL import Image


class ADE20KDataset(torch.utils.data.Dataset):
    def __init__(self, path, subset, transform=None):
        self.data = pd.DataFrame({
            'image': sorted(glob.glob(os.path.join(path, 'images', subset, '*.jpg'))),
            'mask': sorted(glob.glob(os.path.join(path, 'annotations', subset, '*.png'))),
        })
        self.data.index = range(len(self.data))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        row = self.data.loc[i]

        input = {
            'image': Image.open(row['image']).convert('RGB'),
            'mask': Image.open(row['mask']),
        }

        if self.transform is not None:
            input = self.transform(input)

        return input
