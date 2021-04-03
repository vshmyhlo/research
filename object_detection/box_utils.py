import torch
import torchvision


def pairwise(a, b):
    return a.unsqueeze(1), b.unsqueeze(0)


def boxes_to_tl_br(boxes):
    return torch.split(boxes, 2, -1)


def tl_br_to_boxes(tl, br):
    return torch.cat([tl, br], -1)


def boxes_to_centers(boxes):
    tl, br = boxes_to_tl_br(boxes)
    return tl_br_to_centers(tl, br)


def tl_br_to_centers(tl, br):
    return (tl + br) / 2


def boxes_size(boxes):
    tl, br = boxes_to_tl_br(boxes)
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
    a_tl, a_br = boxes_to_tl_br(a)
    b_tl, b_br = boxes_to_tl_br(b)

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
    a_tl, a_br = boxes_to_tl_br(a)
    b_tl, b_br = boxes_to_tl_br(b)

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
# TODO: pass dets


def boxes_to_offsets(boxes, points):
    tl, br = boxes_to_tl_br(boxes)

    offsets = torch.cat(
        [
            points - tl,
            br - points,
        ],
        -1,
    )

    return offsets


def offsets_to_boxes(offsets, points):
    tl, br = boxes_to_tl_br(offsets)

    boxes = torch.cat(
        [
            points - tl,
            points + br,
        ],
        -1,
    )

    return boxes


def boxes_contain_points(boxes, points):
    tl, br = boxes_to_tl_br(boxes)
    mask = (tl < points) & (points < br)
    mask = mask.all(-1)

    return mask
