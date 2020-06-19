import torch

from fcos.utils import Detections, flatten_detection_map
from object_detection.box_utils import boxes_area, boxes_to_offsets, offsets_to_boxes, per_class_nms, pairwise, \
    boxes_contain_points


class BoxCoder(object):
    def __init__(self, levels):
        self.levels = levels
        self.conf_threshold = 0.05
        self.iou_threshold = 0.5
       
    def encode(self, dets, size):
        size = torch.tensor(size, dtype=torch.long)

        strides = []
        class_maps = []
        loc_maps = []
        for i, level in enumerate(self.levels):
            if level is not None:
                stride = 2**i
                strides.append(torch.empty(size.prod(), device=dets.boxes.device).fill_(stride))

                yx_map = build_yx_map(size, stride, device=dets.boxes.device)
                yx_map = flatten_detection_map(yx_map)
                class_map, loc_map = boxes_to_map(dets, yx_map, stride, level)

                class_maps.append(class_map)
                loc_maps.append(loc_map)

            size = torch.ceil(size.float() / 2).long()

        strides = torch.cat(strides, 0)
        class_maps = torch.cat(class_maps, 0)
        loc_maps = torch.cat(loc_maps, 0)

        loc_maps /= strides.unsqueeze(1)

        return class_maps, loc_maps

    def decode(self, class_maps, loc_maps, size):
        size = torch.tensor(size, dtype=torch.long)

        strides = []
        yx_maps = []
        for i, level in enumerate(self.levels):
            if level is not None:
                stride = 2**i
                strides.append(torch.empty(size.prod(), device=class_maps.device).fill_(stride))

                yx_map = build_yx_map(size, stride, device=class_maps.device)
                yx_map = flatten_detection_map(yx_map)

                yx_maps.append(yx_map)

            size = torch.ceil(size.float() / 2).long()

        strides = torch.cat(strides, 0)
        yx_maps = torch.cat(yx_maps, 0)

        loc_maps *= strides.unsqueeze(1)
        loc_maps = offsets_to_boxes(loc_maps, yx_maps)

        scores, class_ids = class_maps.max(1)
        fg = scores > self.conf_threshold

        boxes = loc_maps
        boxes = boxes[fg]
        class_ids = class_ids[fg]
        scores = scores[fg]

        keep = per_class_nms(boxes, scores, class_ids, self.iou_threshold)
        boxes = boxes[keep]
        class_ids = class_ids[keep]
        scores = scores[keep]

        return Detections(
            class_ids=class_ids,
            boxes=boxes,
            scores=scores)


def compute_sub_boxes(boxes, stride):
    return boxes


def offsets_bounded(offsets, bounds):
    max = offsets.max(-1).values
    mask = (bounds[0] < max) & (max < bounds[1])

    return mask


def boxes_to_map(dets, yx_map, stride, bounds):
    if dets.boxes.size(0) == 0:
        class_map = torch.zeros(yx_map.size(0), dtype=torch.long)
        loc_map = torch.zeros(yx_map.size(0), 4, dtype=torch.float)

        return class_map, loc_map

    offsets = boxes_to_offsets(*pairwise(dets.boxes, yx_map))
    sub_boxes = compute_sub_boxes(dets.boxes, stride)
    inside = boxes_contain_points(*pairwise(sub_boxes, yx_map))

    bounded = offsets_bounded(offsets, bounds)
    matches = inside & bounded

    areas = boxes_area(dets.boxes).unsqueeze(1).repeat(1, offsets.size(1))
    areas[~matches] = float('inf')

    indices = areas.argmin(0)

    class_ids = dets.class_ids[indices] + 1
    class_ids[~matches.any(0)] = 0
    offsets = offsets[indices, range(indices.size(0))]

    class_map = class_ids
    loc_map = offsets

    return class_map, loc_map


def build_yx_map(size, stride, device=None):
    y = torch.arange(0, size[0] * stride, stride, dtype=torch.float, device=device)
    x = torch.arange(0, size[1] * stride, stride, dtype=torch.float, device=device)
    yx = torch.meshgrid(y, x)
    yx = torch.stack(yx, 0)
    yx += stride // 2
    assert yx.size() == (2, *size)

    return yx
