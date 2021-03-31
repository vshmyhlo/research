import os

import numpy as np
import pandas as pd
import torch.utils.data
from PIL import Image, UnidentifiedImageError


class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform):
        self.data = pd.DataFrame({"image_path": os.listdir(path)})
        self.data["image_path"] = self.data["image_path"].apply(
            lambda image_path: os.path.join(path, image_path)
        )
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        row = self.data.iloc[i]
        image = Image.open(row["image_path"])
        if self.transform is not None:
            image = self.transform(image)
        return image
