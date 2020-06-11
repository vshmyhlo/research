import os

import numpy as np
import pandas as pd
import torch.utils.data

from model_selection import SimpleStratifiedGroupKFold

FOLDS = 5
SEX_TO_ID = {
    np.nan: 0,
    'male': 1,
    'female': 2,
}
SITE_TO_ID = {
    np.nan: 0,
    'lower extremity': 1,
    'upper extremity': 2,
    'torso': 3,
    'anterior torso': 3,
    'posterior torso': 3,
    'head/neck': 4,
    'palms/soles': 5,
    'oral/genital': 6,
}


class ConcatDataset(torch.utils.data.ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)

        data = pd.concat([ds.data for ds in datasets])
        self.data = data.reset_index(drop=True)


class Dataset2020KFold(torch.utils.data.Dataset):
    def __init__(self, path, train, fold, transform=None):
        assert fold in range(1, FOLDS + 1)

        data = self.load_data(path)

        folds = SimpleStratifiedGroupKFold(FOLDS, shuffle=True, random_state=42) \
            .split(data['target'], data['target'], data['patient_id'])
        train_indices, eval_indices = list(folds)[fold - 1]
        if train:
            data = data.loc[train_indices]
        else:
            data = data.loc[eval_indices]

        self.data = data.reset_index(drop=True)
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
                'sex': np.array(SEX_TO_ID[sample['sex']], dtype=np.int64),
                'site': np.array(SITE_TO_ID[sample['anatom_site_general_challenge']], dtype=np.int64),
            },
            'target': np.array(sample['target'], dtype=np.float32),
        }

        if self.transform is not None:
            input = self.transform(input)

        return input

    @staticmethod
    def load_data(path):
        data = pd.read_csv(os.path.join(path, 'train.csv'))
        data['target'] = data['target'] == 1.
        data['jpeg_path'] = data['image_name'].apply(lambda x: os.path.join(path, 'jpeg', 'train', '{}.jpg'.format(x)))
        data = data.reset_index(drop=True)

        return data


class Dataset2020Test(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        data = self.load_data(path)

        self.data = data.reset_index(drop=True)
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
                'sex': np.array(SEX_TO_ID[sample['sex']], dtype=np.int64),
                'site': np.array(SITE_TO_ID[sample['anatom_site_general_challenge']], dtype=np.int64),
            },
        }

        if self.transform is not None:
            input = self.transform(input)

        return input

    @staticmethod
    def load_data(path):
        data = pd.read_csv(os.path.join(path, 'test.csv'))
        data['jpeg_path'] = data['image_name'].apply(lambda x: os.path.join(path, 'jpeg', 'test', '{}.jpg'.format(x)))
        data = data.reset_index(drop=True)

        return data


class Dataset2019(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        data = self.load_data(path)

        self.data = data.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        sample = self.data.loc[i]

        input = {
            'id': sample['image'],
            'image': sample['jpeg_path'],
            'meta': {
                'age': np.array(sample['age_approx'], dtype=np.float32),
                'sex': np.array(SEX_TO_ID[sample['sex']], dtype=np.int64),
                'site': np.array(SITE_TO_ID[sample['anatom_site_general']], dtype=np.int64),
            },
            'target': np.array(sample['target'], dtype=np.float32),
        }

        if self.transform is not None:
            input = self.transform(input)

        return input

    @staticmethod
    def load_data(path):
        data = pd.read_csv(os.path.join(path, 'ISIC_2019_Training_GroundTruth.csv'))
        meta = pd.read_csv(os.path.join(path, 'ISIC_2019_Training_Metadata.csv'))
        data = data.merge(meta, on='image', how='left')

        data['target'] = data['MEL'] == 1.
        data['jpeg_path'] = data['image'].apply(
            lambda x: os.path.join(path, 'ISIC_2019_Training_Input', 'ISIC_2019_Training_Input', '{}.jpg'.format(x)))
        data = data.reset_index(drop=True)

        return data
