import pytest
import numpy as np


from egsis import metrics


@pytest.fixture
def x():
    return np.array(
        [[0, 0, 1],
         [0, 1, 0],
         [1, 0, 0]]
    )


@pytest.fixture
def y():
    return np.array(
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]]
    )


def test_pixel_accuracy(x, y):
    assert metrics.pixel_accuracy(x, y) == pytest.approx(0.5555, 0.001)


def test_iou(x, y):
    assert metrics.iou(x, y) == 0.20


def test_f1(x, y):
    assert metrics.f1(x, y) == pytest.approx(0.33333, 0.001)


def test_iou_raise_error_on_non_binarized_labels(x, y):
    with pytest.raises(ValueError):
        metrics.iou(x + 1, y + 1)


def test_f1_raise_error_on_non_binarized_labels(x, y):
    with pytest.raises(ValueError):
        metrics.iou(x + 1, y + 1)
