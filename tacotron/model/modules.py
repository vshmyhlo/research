import torch.nn as nn


class ConvNorm1d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, init='linear'):
        super().__init__(
            Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False, init=init),
            nn.BatchNorm1d(out_channels))


class Conv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, init='linear'):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain(init))
        if bias:
            nn.init.zeros_(self.bias)


class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, init='linear'):
        super().__init__(in_features, out_features, bias=bias)

        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain(init))
        if bias:
            nn.init.zeros_(self.bias)
