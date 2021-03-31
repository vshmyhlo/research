import torch
from torch import nn as nn
from torch.nn import functional as F


class BinaryCrossEntropyLoss(nn.Module):
    def forward(self, input, target):
        if target:
            target = torch.ones_like(input)
        else:
            target = torch.zeros_like(input)

        return F.binary_cross_entropy_with_logits(input=input, target=target)


class NonSatLogisticLoss(nn.Module):
    def forward(self, input, target):
        input = F.logsigmoid(input)

        if target:
            return -input
        else:
            return input


class WassersteinLoss(nn.Module):
    def forward(self, input, target):
        if target:
            return -input.mean()
        else:
            return input.mean()


class LogisticNSLoss(nn.Module):
    def forward(self, input, target):
        if target:
            return F.softplus(-input).mean()
        else:
            return F.softplus(input).mean()


class LeastSquaresLoss(nn.Module):
    def forward(self, input, target):
        if target:
            return (input - 1).square().mean()
        else:
            return (input - 0).square().mean()
