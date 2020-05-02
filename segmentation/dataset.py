import os
from pathlib import Path

import pandas as pd
import torch.utils.data
from PIL import Image


def load_labeled_data(_, binary_path):
    def mask_path_to_image_path(mask_path):
        dirname = os.path.dirname(mask_path)
        basename, _ = os.path.splitext(os.path.basename(mask_path))
        basename, _ = basename.split('_')
        return os.path.join(dirname, '{}.jpg'.format(basename))

    data = pd.DataFrame({
        'mask_path': sorted([str(p) for p in Path(binary_path).rglob('*_mz.png')])
    })
    data['image_path'] = data['mask_path'].apply(mask_path_to_image_path)

    return data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        row = self.data.iloc[i]

        input = {
            'image': Image.open(row['image_path']),
            'mask': Image.open(row['mask_path']),
        }

        input = self.transform(input)

        return input
