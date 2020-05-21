import torchvision
from torch import nn as nn
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.net = torchvision.models.resnet50(num_classes=num_classes, pretrained=False)

    def forward(self, input):
        input = self.net(input)
        input = F.softmax(input, 1)

        return input
