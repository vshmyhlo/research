from torch import nn as nn

from tacotron.model.modules import Linear, Conv1d
from tacotron.utils import transpose_t_c


class Attention(nn.Module):
    def __init__(self, query_features, key_features, mid_features):
        super().__init__()

        self.query = Linear(query_features, mid_features, bias=False, init='tanh')
        self.key = Linear(key_features, mid_features, bias=False, init='tanh')
        self.weight = LocationLayer(out_features=mid_features, init='tanh')

        self.scores = nn.Sequential(
            nn.Tanh(),
            Linear(mid_features, 1))

    def forward(self, query, key, value, mask, weight):
        query = self.query(query.unsqueeze(1))
        # key = self.key(key) # FIXME:
        weight = self.weight(weight)

        scores = self.scores(query + key + weight)
        scores = scores.masked_fill_(~mask.unsqueeze(-1), float('-inf'))
        weight = scores.softmax(1)

        context = (weight * value).sum(1)

        return context, weight.squeeze(2)


class LocationLayer(nn.Module):
    def __init__(self, out_features, init='linear'):
        super().__init__()

        self.conv = Conv1d(2, 32, 31, padding=31 // 2, bias=False)
        self.linear = Linear(32, out_features, bias=False, init=init)

    def forward(self, input):
        input = self.conv(input)
        input = transpose_t_c(input)
        input = self.linear(input)

        return input
