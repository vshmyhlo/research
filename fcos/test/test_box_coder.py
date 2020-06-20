import torch

from fcos.box_coder import boxes_to_map, build_yx_map, compute_sub_boxes
from fcos.utils import Detections, flatten_detection_map


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

    yx_map = build_yx_map((4, 4), 32, device=dets.boxes.device)
    yx_map = flatten_detection_map(yx_map)
    class_map_a, _ = boxes_to_map(dets, yx_map, 32, (0, float('inf')))

    class_map_e = torch.tensor([
        [3, 4, 0, 0],
        [0, 4, 5, 0],
        [0, 5, 5, 0],
        [0, 0, 0, 0],
    ], dtype=class_map_a.dtype).view(-1)

    assert torch.allclose(class_map_a, class_map_e)


def test_compute_sub_boxes():
    boxes = torch.tensor([
        [0, 0, 100, 20],
    ], dtype=torch.float)
   
    actual = compute_sub_boxes(boxes, 32)
    expected = torch.tensor([
        [2, 0, 98, 20],
    ], dtype=torch.float)

    assert torch.allclose(actual, expected)
