import pytest
import numpy


@pytest.fixture
def image_a():
    return numpy.array(
        [
            [10, 20, 30],
            [10, 20, 30],
            [10, 20, 30]
        ]
    )


@pytest.fixture
def image_b():
    return numpy.array(
        [
            [30, 20, 10],
            [30, 20, 10],
            [30, 20, 10]
        ]
    )


@pytest.fixture
def segments():
    return numpy.array(
        [
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
         ]
    )
