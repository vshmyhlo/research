import torch

from fcos.utils import foreground_binary_coding
from losses import sigmoid_focal_loss, offsets_iou_loss


def compute_loss(input, target):
    input_class, input_loc = input
    target_class, target_loc = target

    # classification loss
    class_mask = target_class != -1
    class_loss = compute_classification_loss(input=input_class[class_mask], target=target_class[class_mask])

    # localization loss
    loc_mask = target_class > 0
    loc_loss = compute_localization_loss(input=input_loc[loc_mask], target=target_loc[loc_mask])

    assert class_loss.size() == loc_loss.size()
    loss = class_loss + loc_loss

    return loss


def compute_classification_loss(input, target):
    if input.numel() == 0:
        return torch.tensor(0.)

    num_pos = (target > 0).sum().clamp(min=1.)
    target = foreground_binary_coding(target, input.size(1))
    loss = sigmoid_focal_loss(input=input, target=target)
    loss = loss.sum() / num_pos

    return loss


def compute_localization_loss(input, target):
    if input.numel() == 0:
        return torch.tensor(0.)

    loss = offsets_iou_loss(input=input, target=target)
    loss = loss.mean()

    return loss
