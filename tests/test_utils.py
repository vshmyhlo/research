from utils import DataLoaderSlice


def test_data_loader_slice():
    dl = DataLoaderSlice(range(5), 3)

    assert list(dl) == [0, 1, 2]
    assert list(dl) == [3, 4, 0]
    assert list(dl) == [1, 2, 3]
