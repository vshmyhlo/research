import glob
import os

import pandas as pd
import torch
from PIL import Image


class DualViewDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        self.data = pd.DataFrame(
            {
                "path": glob.glob(os.path.join(path, "*.jpg")),
            }
        ).reset_index(drop=True)

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = self.data.iloc[item]
        image1 = image2 = Image.open(sample["path"])

        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2
