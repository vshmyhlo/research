import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm


class StratifiedGroupKFold(object):
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, x, y, groups):
        labels_num = np.max(y) + 1
        y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
        y_distr = Counter()
        for label, g in zip(y, groups):
            y_counts_per_group[g][label] += 1
            y_distr[label] += 1

        y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
        groups_per_fold = defaultdict(set)

        def eval_y_counts_per_fold(y_counts, fold):
            y_counts_per_fold[fold] += y_counts
            std_per_label = []
            for label in range(labels_num):
                label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(self.n_splits)])
                std_per_label.append(label_std)
            y_counts_per_fold[fold] -= y_counts
            return np.mean(std_per_label)

        groups_and_y_counts = list(y_counts_per_group.items())
        random.Random(self.random_state).shuffle(groups_and_y_counts)

        for g, y_counts in tqdm(sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])),
                                total=len(groups_and_y_counts)):
            best_fold = None
            min_eval = None
            for i in range(self.n_splits):
                fold_eval = eval_y_counts_per_fold(y_counts, i)
                if min_eval is None or fold_eval < min_eval:
                    min_eval = fold_eval
                    best_fold = i
            y_counts_per_fold[best_fold] += y_counts
            groups_per_fold[best_fold].add(g)

        all_groups = set(groups)
        for i in range(self.n_splits):
            train_groups = all_groups - groups_per_fold[i]
            test_groups = groups_per_fold[i]

            train_indices = [i for i, g in enumerate(groups) if g in train_groups]
            test_indices = [i for i, g in enumerate(groups) if g in test_groups]

            yield train_indices, test_indices


class SimpleStratifiedGroupKFold(object):
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, x, y, groups):
        def compute_stats(data):
            return pd.Series(
                [(~data['target']).sum(), data['target'].sum()],
                index=['neg', 'pos'])

        data = pd.DataFrame({'target': y, 'group': groups}).reset_index(drop=True)
        groups = data.groupby('group').apply(compute_stats).sort_values('neg')

        data['fold'] = None
        for n in range(self.n_splits):
            data.loc[data['group'].isin(groups.iloc[n::self.n_splits].index), 'fold'] = n

        for n in range(self.n_splits):
            yield data[data['fold'] != n].index.values, data[data['fold'] == n].index.values
