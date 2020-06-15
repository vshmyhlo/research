import torch
import torch.nn.functional as F

from fcos.utils import foreground_binary_coding
from losses import sigmoid_focal_loss


# TODO: add ignored positions
def compute_loss(input, target):
    input_class, input_loc = [flatten_and_concat_maps(x) for x in input]
    target_class, target_loc = [flatten_and_concat_maps(x) for x in target]

    # classification loss
    class_loss = compute_class_loss(input=input_class, target=target_class)

    # localization loss
    loc_mask = target_class > 0
    loc_loss = compute_localization_loss(input=input_loc[loc_mask], target=target_loc[loc_mask])

    assert class_loss.size() == loc_loss.size()
    loss = class_loss + loc_loss

    return loss


def compute_class_loss(input, target):
    num_pos = (target > 0).sum().clamp(min=1.)

    target = foreground_binary_coding(target, input.size(2))
    loss = sigmoid_focal_loss(input=input, target=target)
    loss = loss.sum() / num_pos

    return loss


def compute_localization_loss(input, target):
    if input.numel() == 0:
        return torch.tensor(0.)

    loss = F.smooth_l1_loss(input=input, target=target, reduction='none')
    loss = loss.mean()

    return loss


def flatten_and_concat_maps(maps):
    def flatten(map):
        b, *c, h, w = map.size()
        map = map.view(b, *c, h * w)
        if map.dim() == 3:
            map = map.transpose(-1, -2)

        return map

    maps = [flatten(map) for map in maps]
    maps = torch.cat(maps, 1)

    return maps
