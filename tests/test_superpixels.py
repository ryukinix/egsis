from egsis import superpixels
import numpy


def test_superpixel_centroids():
    example = numpy.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ]
    )
    expected = {
        0: [1, 1],
        1: [1, 1]
    }
    assert superpixels.superpixel_centroids(example) == expected
