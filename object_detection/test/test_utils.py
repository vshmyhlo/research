import torch

from object_detection.utils import per_class_nms


def test_per_class_nms():
    boxes = torch.tensor([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 0.5, 0.5],
    ], dtype=torch.float)
    scores = torch.ones(boxes.size(0), dtype=torch.float)
    class_ids = torch.tensor([0, 0, 0], dtype=torch.long)

    actual = per_class_nms(boxes, scores, class_ids, iou_threshold=0.5)
    expected = torch.tensor([0, 2], dtype=torch.long)
    assert torch.allclose(actual, expected)

    class_ids = torch.tensor([0, 1, 1], dtype=torch.long)

    actual = per_class_nms(boxes, scores, class_ids, iou_threshold=0.5)
    expected = torch.tensor([0, 1, 2], dtype=torch.long)
    assert torch.allclose(actual, expected)
