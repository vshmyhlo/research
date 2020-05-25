import os

import pandas as pd
import torch.utils.data


class LJ(torch.utils.data.Dataset):
    def __init__(self, path, transform):
        self.metadata = pd.read_csv(os.path.join(path, 'metadata.csv'), sep='|', names=['id', 'text', 'text_norm'])
        self.metadata['wav_path'] = \
            self.metadata['id'].apply(lambda id: os.path.join(path, 'wavs', '{}.wav'.format(id)))
        # FIXME:
        for field in ['text', 'text_norm']:
            self.metadata = self.metadata[~self.metadata[field].isna()]
        self.metadata.index = range(len(self.metadata))
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
