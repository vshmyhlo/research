import os

import pandas as pd
import torch.utils.data
from sklearn.model_selection import train_test_split


class LJ(torch.utils.data.Dataset):
    def __init__(self, path, subset, transform):
        metadata = load_metadata(path)
        subsets = {}
        subsets['train'], subsets['test'] = train_test_split(metadata, test_size=0.1, random_state=42)

        self.metadata = subsets[subset].reset_index()
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, i):
        sample = self.metadata.loc[i]

        input = {
            'text': sample['text_norm'],
            'audio': sample['wav_path'],
        }

        if self.transform is not None:
            input = self.transform(input)

        return input


def load_metadata(path):
    metadata = pd.read_csv(os.path.join(path, 'metadata.csv'), sep='|', names=['id', 'text', 'text_norm'])
    metadata['wav_path'] = metadata['id'].apply(lambda id: os.path.join(path, 'wavs', '{}.wav'.format(id)))
    # FIXME:
    for field in ['text', 'text_norm']:
        metadata = metadata[~metadata[field].isna()]

    return metadata
