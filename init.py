import math

import torch.nn as nn


class Normal(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def init_(self, weight):
        nn.init.normal_(weight, self.mean, self.std)


class KaimingUniform(object):
    def __init__(self, nonlinearity="linear"):
        self.nonlinearity = nonlinearity

    def init_(self, weight):
        nn.init.kaiming_uniform_(weight, nonlinearity=self.nonlinearity)


def prior_(tensor, prob):
    nn.init.constant_(tensor, -math.log((1 - prob) / prob))
