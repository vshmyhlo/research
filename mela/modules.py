import torch
from torch import nn as nn

from utils import weighted_sum


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
            nn.Linear(mid_features * 3, out_features),
        )

        for w in [self.age_0, self.age_1, self.sex.weight, self.site.weight]:
            nn.init.normal_(w)
        nn.init.kaiming_normal_(self.output[2].weight, nonlinearity="linear")

    def forward(self, input):
        age = (input["age"] / 100.0).unsqueeze(1)
        age_is_nan = torch.isnan(age)
        age[age_is_nan] = 0.0

        age_0 = torch.where(age_is_nan, self.age_nan, self.age_0)
        age_1 = torch.where(age_is_nan, self.age_nan, self.age_1)
        age = weighted_sum(age_0, age_1, age)

        sex = self.sex(input["sex"])
        site = self.site(input["site"])

        input = torch.cat([age, sex, site], 1)
        input = self.output(input)

        return input
