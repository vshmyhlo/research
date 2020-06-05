import os

import numpy as np
import pandas as pd
import torch.utils.data
from sklearn.model_selection import StratifiedKFold

FOLDS = 5
SEX = [np.nan, 'male', 'female']
SITE = [np.nan, 'lower extremity', 'torso', 'upper extremity', 'head/neck', 'palms/soles', 'oral/genital']

C = ['unk', 'nev', 'ben_ker', 'mel', 'lich_ker', 'lent_nos', 'derm_fib', 'sq_cell_carc', 'bas_cell_carc', 'vasc_les',
     'act_ker', 'at_mel_prol', 'caf_au_lait_mac']


class ConcatDataset(torch.utils.data.ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)

        data = pd.concat([ds.data for ds in datasets])
        self.data = data.reset_index(drop=True)


class Dataset2020(torch.utils.data.Dataset):
    def __init__(self, path, train, fold, transform=None):
        data = self.load_data(path)
        folds = StratifiedKFold(FOLDS, shuffle=True, random_state=42).split(data['target'], data['target'])
        train_indices, eval_indices = list(folds)[fold]
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
                'sex': np.array(SEX.index(sample['sex']), dtype=np.int64),
                'site': np.array(SITE.index(sample['anatom_site_general_challenge']), dtype=np.int64),
            },
            'target': np.array(sample['target'], dtype=np.float32),
            'diag': np.array(C.index(sample['diag']), dtype=np.int64),
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

        diag_mapping = {
            'unknown': 'unk',
            'nevus': 'nev',
            'melanoma': 'mel',
            'seborrheic keratosis': 'ben_ker',
            'lentigo NOS': 'lent_nos',
            'lichenoid keratosis': 'lich_ker',
            'solar lentigo': 'ben_ker',
            'cafe-au-lait macule': 'caf_au_lait_mac',
            'atypical melanocytic proliferation': 'at_mel_prol',
        }
        data['diag'] = data['diagnosis'].apply(lambda x: diag_mapping[x])

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
                'age': np.array(0, dtype=np.float32),
                'sex': np.array(0, dtype=np.int64),
                'site': np.array(0, dtype=np.int64),
            },
            'target': np.array(sample['target'], dtype=np.float32),
            'diag': np.array(C.index(sample['diag']), dtype=np.int64),
        }

        if self.transform is not None:
            input = self.transform(input)

        return input

    @staticmethod
    def load_data(path):
        data = pd.read_csv(os.path.join(path, 'ISIC_2019_Training_GroundTruth.csv'))
        data['jpeg_path'] = data['image'].apply(
            lambda x: os.path.join(path, 'ISIC_2019_Training_Input', 'ISIC_2019_Training_Input', '{}.jpg'.format(x)))
        data = data.reset_index(drop=True)

        diag_mapping = {
            'MEL': 'mel',
            'NV': 'nev',
            'BCC': 'bas_cell_carc',
            'AK': 'act_ker',
            'BKL': 'ben_ker',
            'DF': 'derm_fib',
            'VASC': 'vasc_les',
            'SCC': 'sq_cell_carc',
            'UNK': 'unk',
        }
        one_hot = data[list(diag_mapping.keys())].values
        data['diag'] = pd.Series(one_hot.argmax(1)).apply(lambda x: list(diag_mapping.values())[x])
        data['target'] = data['diag'] == 'mel'

        return data
