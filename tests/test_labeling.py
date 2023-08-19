import pytest

import numpy

from egsis import labeling


@pytest.fixture
def label_matrix():
    return numpy.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
    )


def test_create_label_matrix(label_matrix):
    result = labeling.create_label_matrix(shape=(3, 3))
    assert numpy.array_equal(result, label_matrix)


def test_set_superpixel_label(label_matrix, segments):
    labeling.set_superpixel_label(
        label_matrix=label_matrix,
        superpixels=segments,
        superpixel=1,
        label=1,
    )
    assert label_matrix[1, :].tolist() == [1, 1, 1]


def test_get_superpixel_label(label_matrix, segments):
    labeling.set_superpixel_label(
        label_matrix=label_matrix,
        superpixels=segments,
        superpixel=1,
        label=1,
    )
    label = labeling.get_superpixel_label(
        label_matrix=label_matrix,
        superpixels=segments,
        superpixel=1
    )
    assert label == 1
