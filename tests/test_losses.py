import torch

from losses import offsets_iou_loss


def test_offsets_iou_loss():
    a = torch.tensor(
        [
            [1, 1, 1, 1],
        ]
    )
    b = torch.tensor(
        [
            [0, 0, 1, 1],
        ]
    )

    actual = offsets_iou_loss(a, b, eps=0.0)
    expected = torch.tensor([0.75])

    assert torch.allclose(actual, expected)
