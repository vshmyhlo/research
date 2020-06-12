from itertools import groupby

import efficientnet_pytorch
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


class EfficientNetB0(nn.Module):
    featuremap_depths = [None, 16, 24, 40, 112, 320]

    def __init__(self):
        super().__init__()

        self.model = efficientnet_pytorch.EfficientNet.from_pretrained('efficientnet-b0')

    def forward(self, input):
        """ Returns output of the final convolution layer """

        # Stem
        input = self.model._swish(self.model._bn0(self.model._conv_stem(input)))

        # Blocks
        blocks = []
        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            input = block(input, drop_connect_rate=drop_connect_rate)
            blocks.append(input)

        blocks = groupby(blocks, lambda x: x.size()[2:])
        blocks = [list(group)[-1] for _, group in blocks]
        input = [None] + blocks
        input[0] = input[1] = None  # do not store c0 and c1

        return input
