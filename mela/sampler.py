import math

import numpy as np
import pandas as pd
import torch.utils.data


class BatchSampler(torch.utils.data.Sampler):
    def __init__(self, data, shuffle=False, drop_last=False):
        super().__init__(data)

        self.data = pd.DataFrame({'target': data}, index=list(range(len(data))))
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        neg = list(self.data[~self.data['target']].index)
        pos = list(self.data[self.data['target']].index)
        r = math.floor(len(neg) / len(pos))

        if self.shuffle:
            np.random.shuffle(neg)
            np.random.shuffle(pos)

        batches = []
        while len(pos) > 0:
            batch = [neg.pop() for _ in range(r)]
            batch.append(pos.pop())
            batches.append(batch)
        assert len(pos) == 0

        if not self.drop_last:
            batch = []
            while len(neg) > 0:
                batch.append(neg.pop())
            batches.append(batch)
            assert len(neg) == 0

        assert len(batches) == len(self)

        yield from batches

    def __len__(self):
        size = self.data['target'].sum()
        if not self.drop_last:
            size += 1

        return size
