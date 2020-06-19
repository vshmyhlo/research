import torch

from fcos.box_coder import boxes_to_map
from fcos.utils import Detections


def test_assign_boxes_to_map():
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

    class_map_a, _ = boxes_to_map(dets, (4, 4), 32, (0, float('inf')))

    class_map_e = torch.tensor([
        [3, 4, 0, 0],
        [0, 4, 5, 0],
        [0, 5, 5, 0],
        [0, 0, 0, 0],
    ], dtype=class_map_a.dtype)

    assert (class_map_a == class_map_e).all()
