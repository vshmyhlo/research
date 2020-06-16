import math

import torch


# TODO: remove explicit torch.tensor(...) calls
# TODO: test
# TODO: check that network predicts same order of boxes
# TODO: rename all anchors maps to anchors
def arrange_anchors_on_grid(image_size, anchor_levels):
    map_size = image_size

    anchor_maps = []
    for anchors in anchor_levels:
        if anchors is not None:
            level_anchor_maps = []
            for anchor in anchors:
                anchor_map = arrange_anchor_on_grid(image_size, map_size, anchor)
                level_anchor_maps.append(anchor_map)

            level_anchor_maps = torch.cat(level_anchor_maps, 0)
            level_anchor_maps = flatten_detection_map(level_anchor_maps, len(anchors))
            anchor_maps.append(level_anchor_maps)

        map_size = torch.ceil(map_size.float() / 2).long()

    anchor_maps = torch.cat(anchor_maps, 0)

    return anchor_maps


# TODO: use helpers
def arrange_anchor_on_grid(image_size, map_size, anchor):
    cell_size = image_size / map_size

    y = torch.linspace(cell_size[0] / 2, image_size[0] - cell_size[0] / 2, map_size[0])
    x = torch.linspace(cell_size[1] / 2, image_size[1] - cell_size[1] / 2, map_size[1])
    y, x = torch.meshgrid(y, x)
    yx = torch.stack((y, x), 0)
    del y, x

    anchor = anchor.view(2, 1, 1)
    tl = yx - anchor / 2
    br = yx + anchor / 2

    anchor_map = torch.cat((tl, br), 0)

    return anchor_map


def compute_anchor(size, ratio, scale):
    h = math.sqrt(size**2 / ratio) * scale
    w = h * ratio

    return torch.tensor((h, w))


def flatten_detection_map(input, num_anchors):
    *rest, c, h, w = input.size()
    n_rest = len(rest)
    input = input.permute(*range(n_rest), n_rest + 1, n_rest + 2, n_rest)
    input = input.reshape(*rest, h * w * num_anchors, c // num_anchors)

    return input
