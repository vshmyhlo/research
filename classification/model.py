import torchvision
from torch import nn as nn


class Model(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()

        if model == 'resnet50':
            self.net = torchvision.models.resnet50(num_classes=num_classes, pretrained=False)
        else:
            raise AssertionError('invalid model {}'.format(model))

    def forward(self, input):
        input = self.net(input)

        return input
