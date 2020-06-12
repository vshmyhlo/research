from torch.nn import functional as F

from detection.box_utils import boxes_area, boxes_intersection
from detection.utils import foreground_binary_coding


def smooth_l1_loss(input, target):
    loss = F.smooth_l1_loss(input=input, target=target, reduction='none')

    return loss


def boxes_iou_loss(input, target):
    intersection = boxes_intersection(input, target)
    union = boxes_area(input) + boxes_area(target) - intersection
    iou = intersection / union
    loss = 1 - iou

    return loss


def focal_loss(input, target, gamma=2., alpha=0.25):
    norm = (target > 0).sum().clamp(min=1.)

    target = foreground_binary_coding(target, input.size(1))

    prob = input.sigmoid()
    prob_true = prob * target + (1 - prob) * (1 - target)
    alpha = alpha * target + (1 - alpha) * (1 - target)
    weight = alpha * (1 - prob_true)**gamma

    loss = F.binary_cross_entropy_with_logits(input=input, target=target, reduction='none')
    loss = (weight * loss).sum() / norm

    return loss
