import math
from itertools import chain

import numpy as np
import torch.utils.data


class BalancedSampler(torch.utils.data.Sampler):
    def __init__(self, data, shuffle=False, drop_last=False):
        super().__init__(data)

        self.data = data
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        neg = list(self.data[~self.data].index)
        pos = list(self.data[self.data].index)
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

        if self.drop_last:
            batches = tmp(batches)

        if not self.drop_last:
            batch = []
            while len(neg) > 0:
                batch.append(neg.pop())
            batches.append(batch)
            assert len(neg) == 0

        assert len(batches) == len(self)

        yield from batches

    def __len__(self):
        size = self.data.sum()
        if not self.drop_last:
            size += 1
        else:
            size = math.floor(size / 5)

        return size


def tmp(batches):
    batches = batches[:math.floor(len(batches) / 5) * 5]
    batches = [
        list(chain(*[batches[i + j] for j in range(5)]))
        for i in range(0, len(batches), 5)]

    return batches
