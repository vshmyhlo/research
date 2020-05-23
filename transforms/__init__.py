import torch


class ApplyTo(object):
    def __init__(self, to, transform):
        self.to = to
        self.transform = transform

    def __call__(self, input):
        input = {
            **input,
            self.to: self.transform(input[self.to])
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
