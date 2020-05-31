import torchvision
from torch import nn as nn


class Model(nn.Module):
    def __init__(self, model):
        super().__init__()

        if model == 'resnet50':
            self.net = torchvision.models.resnet50(pretrained=True)
        else:
            raise AssertionError('invalid model {}'.format(model))

        self.net.fc = nn.Linear(self.net.fc.in_features, 1)

    def forward(self, input):
        input = self.net(input)
        input = input.squeeze(1)

        return input
