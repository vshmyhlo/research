import torch

from detection.box_utils import boxes_pairwise_iou, boxes_center, boxes_size, per_class_nms
from detection.utils import Detections


def boxes_to_shifts_scales(boxes, anchors):
    shifts = (boxes_center(boxes) - boxes_center(anchors)) / boxes_size(anchors)
    scales = boxes_size(boxes) / boxes_size(anchors)
    shifts_scales = torch.cat([shifts, scales.log()], -1)

    return shifts_scales


def shifts_scales_to_boxes(shifts_scales, anchors):
    shifts, scales = torch.split(shifts_scales, 2, -1)
    centers = shifts * boxes_size(anchors) + boxes_center(anchors)
    sizes = scales.exp() * boxes_size(anchors)
    boxes = torch.cat([centers - sizes / 2, centers + sizes / 2], -1)

    return boxes


# TODO: encode and decode has different class indexing
# TODO: rename encode/decode with anchors
def encode_boxes(detections, anchors, min_iou, max_iou):
    if detections.boxes.size(0) == 0:
        class_output = torch.zeros(anchors.size(0), dtype=torch.long)
        loc_output = torch.zeros(anchors.size(0), 4, dtype=torch.float)

        return class_output, loc_output

    ious = boxes_pairwise_iou(detections.boxes, anchors)
    iou_values, iou_indices = ious.max(0)

    # build class_output
    class_output = detections.class_ids[iou_indices] + 1
    class_output[iou_values < min_iou] = 0
    class_output[(min_iou <= iou_values) & (iou_values <= max_iou)] = -1

    # build loc_output
    loc_output = detections.boxes[iou_indices]

    return class_output, loc_output


def decode_boxes(input):
    class_output, loc_output = input

    scores, class_ids = class_output.max(1)
    fg = scores > 0.5

    boxes = loc_output
    boxes = boxes[fg]
    class_ids = class_ids[fg]
    scores = scores[fg]

    keep = per_class_nms(boxes, scores, class_ids, 0.5)
    boxes = boxes[keep]
    class_ids = class_ids[keep]
    scores = scores[keep]

    return Detections(
        class_ids=class_ids,
        boxes=boxes,
        scores=scores)
