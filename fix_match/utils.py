class UDataset(object):
    def __init__(self, *datasets):
        assert all(len(datasets[0]) == len(dataset) for dataset in datasets)

        self.datasets = datasets

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, i):
        return tuple(dataset[i] for dataset in self.datasets)


class XUDataLoader(object):
    def __init__(self, *data_loaders):
        self.data_loaders = data_loaders

    def __len__(self):
        return min(map(len, self.data_loaders))

    def __iter__(self):
        for (x_w_images, x_targets), ((u_w_images, _), (u_s_images, _)) in zip(*self.data_loaders):
            yield (x_w_images, x_targets), (u_w_images, u_s_images)
