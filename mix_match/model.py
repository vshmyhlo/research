import torchvision
from torch import nn as nn
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()

        if model == "resnet34":
            self.net = torchvision.models.resnet34(num_classes=num_classes, pretrained=False)
        elif model == "resnet50":
            self.net = torchvision.models.resnet50(num_classes=num_classes, pretrained=False)
        else:
            raise AssertionError("invalid model {}".format(model))

    def forward(self, input):
        input = self.net(input)
        input = F.softmax(input, 1)

        return input
