import torchvision
from torch import nn as nn


class Model(nn.Module):
    def __init__(self, model):
        super().__init__()

        if model == 'resnet50':
            self.net = torchvision.models.resnet50(num_classes=1, pretrained=True)
        else:
            raise AssertionError('invalid model {}'.format(model))

    def forward(self, input):
        input = self.net(input)
        print(input.shape)
        input = self.net[:, 1]
        print(input.shape)
        fali

        return input
