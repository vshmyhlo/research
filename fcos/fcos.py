import torch

from fcos.utils import Detections
from fcos.utils import foreground_binary_coding
from object_detection.box_utils import boxes_area, boxes_pairwise_offsets, boxes_offsets_to_tlbr
from object_detection.box_utils import per_class_nms


class BoxCoder(object):
    def __init__(self, levels):
        self.levels = levels

    def encode(self, dets, size):
        size = torch.tensor(size, dtype=torch.long)

        class_maps = []
        loc_maps = []
        for i, level in enumerate(self.levels):
            if level is not None:
                stride = 2**i

                class_map, loc_map = boxes_to_map(dets, size, stride, level)
                loc_map = loc_map / stride

                class_maps.append(class_map)
                loc_maps.append(loc_map)

            size = torch.ceil(size.float() / 2).long()

        return class_maps, loc_maps

    def decode(self, class_maps, loc_maps, size):
        size = torch.tensor(size, dtype=torch.long)

        class_maps = iter(class_maps)
        loc_maps = iter(loc_maps)

        boxes = []
        scores = []
        for i, level in enumerate(self.levels):
            if level is not None:
                stride = 2**i

                class_map = next(class_maps)
                loc_map = next(loc_maps)

                if class_map.dim() == 2:
                    class_map = foreground_binary_coding(class_map, 80).permute(2, 0, 1)
                else:
                    class_map = class_map.sigmoid()
                loc_map = loc_map * stride
                b, s = map_to_boxes(class_map, loc_map, size, stride)

                boxes.append(b)
                scores.append(s)

            size = torch.ceil(size.float() / 2).long()

        boxes = torch.cat(boxes, 0)
        scores = torch.cat(scores, 0)

        scores, class_ids = scores.max(1)
        fg = scores > 0.5

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


def boxes_to_map(dets, size, stride, bounds):
    if dets.boxes.size(0) == 0:
        class_map = torch.zeros(*size, dtype=torch.long)
        loc_map = torch.zeros(4, *size, dtype=torch.float)

        return class_map, loc_map

    yx_map = build_yx_map(size, stride, device=dets.boxes.device)
    yx_map = yx_map.view(2, size[0] * size[1]).transpose(0, 1)

    offsets = boxes_pairwise_offsets(dets.boxes, yx_map)
    offsets_min = offsets.min(-1).values
    offsets_max = offsets.max(-1).values

    contains = 0 < offsets_min
    limited = (bounds[0] <= offsets_max) & (offsets_max <= bounds[1])
    matches = contains & limited

    areas = boxes_area(dets.boxes).unsqueeze(1).repeat(1, offsets.size(1))
    areas[~matches] = float('inf')

    indices = areas.argmin(0)

    class_ids = dets.class_ids[indices] + 1
    class_ids[~matches.any(0)] = 0
    offsets = offsets[indices, range(indices.size(0))]

    class_map = class_ids.view(*size)
    loc_map = offsets.transpose(0, 1).view(4, *size)

    return class_map, loc_map


def map_to_boxes(class_map, loc_map, size, stride):
    scores = class_map.view(80, size[0] * size[1]).transpose(0, 1)
    offsets = loc_map.view(4, size[0] * size[1]).transpose(0, 1)

    yx_map = build_yx_map(size, stride, device=class_map.device)
    yx_map = yx_map.view(2, size[0] * size[1]).transpose(0, 1)

    boxes = boxes_offsets_to_tlbr(offsets, yx_map)

    return boxes, scores


def build_yx_map(size, stride, device=None):
    y = torch.arange(0, size[0] * stride, stride, dtype=torch.float, device=device)
    x = torch.arange(0, size[1] * stride, stride, dtype=torch.float, device=device)
    yx = torch.meshgrid(y, x)
    yx = torch.stack(yx, 0)
    yx += stride // 2
    assert yx.size() == (2, *size)

    return yx
