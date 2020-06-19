import torch
from torch.nn import functional as F

from object_detection.box_utils import boxes_tl_br


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


def offsets_iou_loss(input, target, eps=1e-7):
    input_tl, input_br = boxes_tl_br(input)
    target_tl, target_br = boxes_tl_br(target)

    input_a = (input_tl + input_br).prod(-1)
    target_a = (target_tl + target_br).prod(-1)

    intersection = torch.min(input_tl, target_tl) + \
                   torch.min(input_br, target_br)
    intersection = intersection.prod(-1)

    union = input_a + target_a - intersection

    iou = intersection / (union + eps)
    loss = 1 - iou

    return loss
