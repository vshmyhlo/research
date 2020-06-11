import torch


class ApplyTo(object):
    def __init__(self, key, transform):
        self.key = key
        self.transform = transform

    def __call__(self, input):
        input = {
            **input,
            self.key: self.transform(input[self.key])
        }

        return input


class Extract(object):
    def __init__(self, fields):
        self.fields = fields

    def __call__(self, input):
        input = tuple(input[k] for k in self.fields)

        return input


class ToTorch(object):
    def __call__(self, input):
        input = torch.tensor(input)

        return input


class Map(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, input):
        input = [self.transform(x) for x in input]

        return input


class Resettable(object):
    def __init__(self, build_transform):
        self.build_transform = build_transform
        self.transform = None

    def __call__(self, input):
        assert self.transform is not None, 'transform is not initialized'

        return self.transform(input)

    def reset(self, *args, **kwargs):
        self.transform = self.build_transform(*args, **kwargs)
