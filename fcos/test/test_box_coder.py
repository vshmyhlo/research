import torch

from fcos.box_coder import boxes_to_map, build_yx_map, compute_sub_boxes, BoxCoder
from fcos.utils import Detections, flatten_detection_map, foreground_binary_coding


def test_box_coder():
    levels = [
        (0, float('inf')),
    ]
    box_coder = BoxCoder(levels)
    size = (128, 128)

    expected = Detections(
        boxes=torch.tensor([
            [0, 0, 32, 32],
            [0, 32, 64, 64],
            [32, 32, 96, 96],
        ], dtype=torch.float),
        class_ids=torch.tensor([
            2,
            3,
            4,
        ], dtype=torch.long),
        scores=None)

    class_maps, loc_maps, cent_maps = box_coder.encode(expected, size)
    class_maps = foreground_binary_coding(class_maps, 10)
    actual = box_coder.decode(class_maps, loc_maps, cent_maps, size)

    assert torch.allclose(actual.boxes, expected.boxes)
    assert torch.allclose(actual.class_ids, expected.class_ids)
    assert torch.allclose(actual.scores, torch.ones_like(actual.scores))


def test_boxes_to_map():
    dets = Detections(
        boxes=torch.tensor([
            [0, 0, 32, 32],
            [0, 32, 64, 64],
            [32, 32, 96, 96],
        ], dtype=torch.float),
        class_ids=torch.tensor([
            2,
            3,
            4,
        ], dtype=torch.long),
        scores=None)

    yx_map = build_yx_map((4, 4), 32)
    yx_map = flatten_detection_map(yx_map)
    class_map_a, _ = boxes_to_map(dets, yx_map, 32, (0, float('inf')))

    class_map_e = torch.tensor([
        [3, 4, 0, 0],
        [0, 4, 5, 0],
        [0, 5, 5, 0],
        [0, 0, 0, 0],
    ], dtype=class_map_a.dtype).view(-1)

    assert torch.allclose(class_map_a, class_map_e)


def test_build_yx_map():
    actual = build_yx_map((2, 2), 32)
    expected = torch.tensor([
        [
            [16, 16],
            [48, 48],
        ], [
            [16, 48],
            [16, 48],
        ]
    ], dtype=torch.float)

    assert torch.allclose(actual, expected)


def test_compute_sub_boxes():
    boxes = torch.tensor([
        [0, 0, 100, 20],
    ], dtype=torch.float)

    actual = compute_sub_boxes(boxes, 20)
    expected = torch.tensor([
        [20, 0, 80, 20],
    ], dtype=torch.float)

    assert torch.allclose(actual, expected)
