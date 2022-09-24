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


def test_superpixel_max_radius():
    example = numpy.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 1],
            [0, 1, 1, 1],
            [0, 1, 1, 1],
        ]
    )
    expected = 2
    centroids = superpixels.superpixel_centroids(example)
    assert superpixels.superpixels_max_radius(example, centroids) == expected
