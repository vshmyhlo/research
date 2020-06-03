import torch
from efficientnet_pytorch import EfficientNet
from torch import nn as nn

from utils import weighted_sum


class Model(nn.Module):
    def __init__(self, model):
        super().__init__()

        if model == 'effnet-b0':
            self.net = EfficientNet.from_pretrained('efficientnet-b0')
        else:
            raise AssertionError('invalid model {}'.format(model))

        self.net._fc = nn.Linear(self.net._fc.in_features, 1)
        self.meta = Meta(128, self.net._fc.in_features)
        self.output = nn.Sequential(
            self.net._dropout,
            self.net._fc)
        self.net._dropout = self.net._fc = nn.Identity()

    def forward(self, input, meta):
        input = self.net(input) + self.meta(meta)
        input = self.output(input)
        input = input.squeeze(1)

        return input


class Meta(nn.Module):
    def __init__(self, mid_features, out_features):
        super().__init__()

        self.age_0 = nn.Parameter(torch.FloatTensor(1, mid_features))
        self.age_1 = nn.Parameter(torch.FloatTensor(1, mid_features))
        self.age_nan = nn.Parameter(torch.FloatTensor(1, mid_features))

        self.sex = nn.Embedding(3, mid_features)
        self.site = nn.Embedding(7, mid_features)

        self.output = nn.Sequential(
            nn.BatchNorm1d(mid_features * 3),
            nn.ReLU(inplace=True),
            nn.Linear(mid_features * 3, out_features))

        for w in [self.age_0, self.age_1, self.sex.weight, self.site.weight]:
            nn.init.normal_(w)
        nn.init.kaiming_normal_(self.output[2].weight, nonlinearity='linear')

    def forward(self, input):
        age = (input['age'] / 100.).unsqueeze(1)
        age_is_nan = torch.isnan(age)
        age = weighted_sum(self.age_0, self.age_1, age)
        age = torch.where(age_is_nan, self.age_nan, age)
       
        sex = self.sex(input['sex'])
        site = self.site(input['site'])

        input = torch.cat([age, sex, site], 1)
        input = self.output(input)

        return input
