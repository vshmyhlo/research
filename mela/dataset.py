import os

import numpy as np
import pandas as pd
import torch.utils.data
from sklearn.model_selection import StratifiedKFold

FOLDS = 5
SEX = [np.nan, 'male', 'female']
SITE = [np.nan, 'lower extremity', 'torso', 'upper extremity', 'head/neck', 'palms/soles', 'oral/genital']


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, train, fold, transform=None):
        data = load_data(path)
        folds = StratifiedKFold(FOLDS, shuffle=True, random_state=42).split(data['target'], data['target'])
        train_indices, eval_indices = list(folds)[fold]
        if train:
            data = data.loc[train_indices].reset_index()
        else:
            data = data.loc[eval_indices].reset_index()

        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        sample = self.data.loc[i]

        input = {
            'id': sample['image_name'],
            'image': sample['jpeg_path'],
            'meta': {
                'age': np.array(sample['age_approx'], dtype=np.float32),
                'sex': np.array(SEX.index(sample['sex']), dtype=np.int64),
                'site': np.array(SITE.index(sample['anatom_site_general_challenge']), dtype=np.int64),
            },
            'target': np.array(sample['target'], dtype=np.float32),
        }

        if self.transform is not None:
            input = self.transform(input)

        return input


def load_data(path):
    data = pd.read_csv(os.path.join(path, 'train.csv'), dtype={'target': bool}).reset_index()
    data['dicom_path'] = data['image_name'].apply(lambda x: os.path.join(path, 'train', '{}.dcm'.format(x)))
    data['jpeg_path'] = data['image_name'].apply(lambda x: os.path.join(path, 'jpeg', 'train', '{}.jpg'.format(x)))

    return data
