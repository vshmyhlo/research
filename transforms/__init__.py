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
