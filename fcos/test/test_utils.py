import torch

from fcos.utils import foreground_binary_coding


def test_foreground_binary_coding():
    input = torch.tensor([
        -1, 0, 1, 2, 3
    ], dtype=torch.long)

    actual = foreground_binary_coding(input, 3)
    expected = torch.tensor([
        [0, 0, 0],
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ], dtype=torch.float)

    assert torch.allclose(actual, expected)
