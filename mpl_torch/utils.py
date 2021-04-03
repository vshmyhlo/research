class XUDataLoader(object):
    def __init__(self, *data_loaders):
        self.data_loaders = data_loaders

    def __len__(self):
        return min(map(len, self.data_loaders))

    def __iter__(self):
        for (x_images, x_targets), (u_images, _) in zip(*self.data_loaders):
            yield (x_images, x_targets), (u_images,)
