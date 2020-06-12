import torch

from fcos.box_utils import boxes_tl_br, boxes_area
import torch

from fcos.box_utils import boxes_tl_br, boxes_area


def build_yx_map(size, stride, device=None):
    y = torch.arange(0, size[0] * stride, stride, dtype=torch.float, device=device)
    x = torch.arange(0, size[1] * stride, stride, dtype=torch.float, device=device)
    yx = torch.meshgrid(y, x)
    yx = torch.stack(yx, -1)
    yx += stride // 2
    assert yx.size() == (*size, 2)

    return yx


def boxes_pairwise_offsets(boxes, points):
    boxes = boxes.unsqueeze(1)
    points = points.unsqueeze(0)

    tl, br = boxes_tl_br(boxes)

    offsets = torch.cat([
        points - tl,
        br - points,
    ], -1)

    return offsets


def map_boxes_to_image(dets, size, stride, bounds):
    yx_map = build_yx_map(size, stride, device=dets.boxes.device)
    yx_map = yx_map.view(size[0] * size[1], 2)

    offsets = boxes_pairwise_offsets(dets.boxes, yx_map)
    offsets_min = offsets.min(-1).values
    offsets_max = offsets.max(-1).values

    contains = 0 < offsets_min
    limited = (bounds[0] <= offsets_max) & (offsets_max <= bounds[1])

    matches = contains & limited
    # print(matches.sum(0).view(*size))

    # ambig = matches.sum(0) > 1
    # print(ambig.long().view(*size))

    areas = boxes_area(dets.boxes).unsqueeze(1).repeat(1, offsets.size(1))
    areas[~matches] = float('inf')
    # print(matches.shape, areas.shape)

    indices = areas.argmin(0)
    # print(offsets.shape, indices.shape)

    class_ids = dets.class_ids[indices] + 1
    class_ids[~matches.any(0)] = 0
    offsets = offsets[indices, range(indices.size(0))]

    class_map = class_ids.view(*size)
    loc_map = offsets.view(*size, 4)

    return class_map, loc_map
