import torchvision
from torch import nn as nn


class Model(nn.Module):
    def __init__(self, num_classes, dropout):
        super().__init__()

        self.net = torchvision.models.resnet18(num_classes=num_classes, pretrained=False)
        self.net.fc = nn.Sequential(
            nn.Dropout(dropout),
            self.net.fc)
       
    def forward(self, input):
        input = self.net(input)
        input = input.softmax(-1)

        return input
