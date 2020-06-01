import torch.nn.functional as F


def sigmoid_cross_entropy(input, target):
    loss = F.binary_cross_entropy_with_logits(input=input, target=target, reduction='none')

    return loss


def sigmoid_focal_loss(input, target, gamma=2.):
    prob = input.sigmoid()
    prob_true = prob * target + (1 - prob) * (1 - target)
    weight = (1 - prob_true)**gamma

    loss = weight * sigmoid_cross_entropy(input=input, target=target)

    return loss
