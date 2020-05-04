class Resettable(object):
    def __init__(self, build_transform):
        self.build_transform = build_transform
        self.transform = None

    def __call__(self, input):
        assert self.transform is not None, 'transform is not initialized'

        return self.transform(input)

    def reset(self, *args, **kwargs):
        self.transform = self.build_transform(*args, **kwargs)
