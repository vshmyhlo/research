import torch

from fcos.utils import foreground_binary_coding
from losses import offsets_iou_loss, sigmoid_cross_entropy, sigmoid_focal_loss


def compute_loss(input, target):
    input_class, input_loc, input_cent = input
    target_class, target_loc, target_cent = target

    num_pos = (target_class > 0).float().sum().clamp(min=1.0)

    # classification loss
    class_mask = target_class != -1
    class_loss = compute_classification_loss(
        input=input_class[class_mask], target=target_class[class_mask]
    )
    class_loss /= num_pos

    # localization loss
    loc_mask = target_class > 0
    loc_loss = compute_localization_loss(input=input_loc[loc_mask], target=target_loc[loc_mask])
    loc_loss /= num_pos

    # centerness loss
    cent_mask = target_class > 0
    cent_loss = compute_centerness_loss(input=input_cent[cent_mask], target=target_cent[cent_mask])
    cent_loss /= num_pos

    assert class_loss.size() == loc_loss.size() == cent_loss.size()

    return {
        "class": class_loss,
        "loc": loc_loss,
        "cent": cent_loss,
    }


def compute_classification_loss(input, target):
    if input.numel() == 0:
        return torch.tensor(0.0)

    target = foreground_binary_coding(target, input.size(1))
    loss = sigmoid_focal_loss(input=input, target=target)
    loss = loss.sum()

    return loss


def compute_localization_loss(input, target):
    if input.numel() == 0:
        return torch.tensor(0.0)

    loss = offsets_iou_loss(input=input, target=target)
    loss = loss.sum()

    return loss


def compute_centerness_loss(input, target):
    if input.numel() == 0:
        return torch.tensor(0.0)

    loss = sigmoid_cross_entropy(input=input, target=target)
    loss = loss.sum()

    return loss
