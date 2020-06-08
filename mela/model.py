from efficientnet_pytorch import EfficientNet
from torch import nn as nn


class Model(nn.Module):
    def __init__(self, model):
        super().__init__()

        if model == 'effnet-b0':
            self.net = EfficientNet.from_pretrained('efficientnet-b0')
        elif model == 'effnet-b1':
            self.net = EfficientNet.from_pretrained('efficientnet-b1')
        else:
            raise AssertionError('invalid model {}'.format(model))

        self.net._fc = nn.Linear(self.net._fc.in_features, 1)
        self.output = nn.Identity()

        # self.net._fc = nn.Linear(self.net._fc.in_features, 1)
        # self.meta = Meta(128, self.net._fc.in_features)
        # self.output = nn.Sequential(
        #     self.net._dropout,
        #     self.net._fc)
        # self.net._dropout = self.net._fc = nn.Identity()

    def forward(self, input, meta):
        input = self.net(input)  # + self.meta(meta)
        input = self.output(input)
        input = input.squeeze(1)

        return input
