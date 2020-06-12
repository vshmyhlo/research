import torchvision
from torch import nn as nn


class ResNet50(nn.Module):
    featuremap_depths = [None, 64, 256, 512, 1024, 2048]

    def __init__(self):
        super().__init__()

        self.model = torchvision.models.resnet50(pretrained=True)

    def forward(self, input):
        input = self.model.conv1(input)
        input = self.model.bn1(input)
        input = self.model.relu(input)
        c1 = input
        input = self.model.maxpool(input)
        input = self.model.layer1(input)
        c2 = input
        input = self.model.layer2(input)
        c3 = input
        input = self.model.layer3(input)
        c4 = input
        input = self.model.layer4(input)
        c5 = input

        input = [None, c1, c2, c3, c4, c5]
        input[0] = input[1] = None  # do not store c0 and c1

        return input
