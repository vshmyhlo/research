from torch.nn import functional as F


def sigmoid_cross_entropy(input, target):
    loss = F.binary_cross_entropy_with_logits(input=input, target=target, reduction='none')

    return loss


def sigmoid_focal_loss(input, target, gamma=2.):
    prob = input.sigmoid()
    prob_true = prob * target + (1 - prob) * (1 - target)
    weight = (1 - prob_true)**gamma

    loss = weight * sigmoid_cross_entropy(input=input, target=target)

    return loss


def f1_loss(input, target, dim=-1, eps=1e-8):
    tp = (input * target).sum(dim)
    fp = (input * (1 - target)).sum(dim)
    fn = ((1 - input) * target).sum(dim)

    f1 = 2 * tp / (2 * tp + fn + fp + eps)
    loss = 1 - f1

    return loss


def lsep_loss(input, target):
    pos_examples = input[target > 0.5].unsqueeze(1)
    neg_examples = input[target <= 0.5].unsqueeze(0)

    loss = F.softplus(neg_examples - pos_examples)

    return loss
