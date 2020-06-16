import torch
import torchvision


def boxes_tl_br(boxes):
    return torch.split(boxes, 2, -1)


def boxes_center(boxes):
    tl, br = boxes_tl_br(boxes)
    return (tl + br) / 2


def boxes_size(boxes):
    tl, br = boxes_tl_br(boxes)
    return br - tl


# TODO: test
def boxes_aspect_ratio(boxes):
    h, w = torch.unbind(boxes_size(boxes), -1)
    return w / h


def boxes_pairwise_iou(a, b):
    iou = torchvision.ops.box_iou(a, b)

    return iou


def boxes_area(boxes):
    hw = boxes_size(boxes)
    area = torch.prod(hw, -1)

    return area


# TODO: test
def boxes_intersection(a, b):
    a_tl, a_br = boxes_tl_br(a)
    b_tl, b_br = boxes_tl_br(b)

    inner_tl = torch.max(a_tl, b_tl)
    inner_br = torch.min(a_br, b_br)
    inner_size = torch.clamp(inner_br - inner_tl, min=0)
    intersection = torch.prod(inner_size, -1)

    return intersection


# TODO: test
def boxes_iou(a, b):
    intersection = boxes_intersection(a, b)
    union = boxes_area(a) + boxes_area(b) - intersection
    iou = intersection / union

    return iou


# TODO: test
def boxes_outer(a, b):
    a_tl, a_br = boxes_tl_br(a)
    b_tl, b_br = boxes_tl_br(b)

    outer_tl = torch.min(a_tl, b_tl)
    outer_br = torch.max(a_br, b_br)
    outer = torch.cat([outer_tl, outer_br], -1)

    return outer


def boxes_clip(boxes, hw):
    boxes = boxes.clone()
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, hw[0])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, hw[1])

    return boxes


# TODO: test
def per_class_nms(boxes, scores, class_ids, iou_threshold):
    mask = torch.zeros(boxes.size(0), dtype=torch.bool)

    for id in class_ids.unique():
        subset_mask = class_ids == id
        keep_mask = torch.zeros(subset_mask.sum(), dtype=torch.bool)

        keep = torchvision.ops.nms(
            boxes[subset_mask],
            scores[subset_mask],
            iou_threshold)

        keep_mask[keep] = True
        mask[subset_mask] = keep_mask

    keep, = torch.where(mask)

    return keep
