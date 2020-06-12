import torch

from fcos.utils import foreground_binary_coding
from losses import sigmoid_focal_loss


def compute_loss(input, target, strides):
    input_class, input_loc = [flatten_and_concat_maps(x) for x in input]
    target_class, target_loc = [flatten_and_concat_maps(x) for x in target]
    print([s.shape for s in strides])
    fail

    # classification loss
    class_loss = compute_class_loss(input=input_class, target=target_class)

    # localization loss
    loc_mask = target_class > 0
    print(loc_mask.mean())
    fail

    loc_loss = compute_localization_loss(input=input_loc[loc_mask], target=target_loc[loc_mask])

    assert class_loss.size() == loc_loss.size()
    loss = class_loss + loc_loss

    return loss


def compute_class_loss(input, target):
    num_pos = (target > 0).sum().clamp(min=1.)

    target = foreground_binary_coding(target, input.size(1))
    loss = sigmoid_focal_loss(input=input, target=target)
    loss = loss.sum() / num_pos

    return loss


def compute_localization_loss(input, target, anchors):
    if input.numel() == 0:
        return torch.tensor(0.)

    if config.loss.localization == 'smooth_l1':
        target = boxes_to_shifts_scales(target, anchors)
        loss = smooth_l1_loss(input=input, target=target)
    elif config.loss.localization == 'iou':
        input = shifts_scales_to_boxes(input, anchors)
        loss = boxes_iou_loss(input=input, target=target)
    else:
        raise AssertionError('invalid config.loss.localization {}'.format(config.loss.localization))

    return loss.mean()


def flatten_and_concat_maps(maps):
    def flatten(map):
        b, *c, h, w = map.size()
        map = map.view(b, *c, h * w)

        return map

    maps = [flatten(map) for map in maps]
    maps = torch.cat(maps, -1)

    return maps
