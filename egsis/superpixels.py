from typing import Dict, List

import numpy

from skimage.segmentation import slic

build_superpixels_from_image = slic


def _calculate_centroid(points: List[numpy.ndarray]) -> List[int]:
    centroid: numpy.ndarray = sum(points)/len(points)  # type: ignore
    return centroid.round().astype(int).tolist()


def superpixel_centroids(segments: numpy.ndarray) -> Dict[int, List[int]]:
    """Computer the centrois of each superpixel

    Parameters
    ----------
    segments: numpy.array
        Segments of image as superpixel

    Return
    ------
    Dictionary with keys as superpixel label and values as (x, y)
    point of centroid.

    """
    superpixel_points: Dict[int, List[numpy.ndarray]] = {}
    height, width = segments.shape
    for y in range(height):
        for x in range(width):
            superpixel = segments[y][x]
            point = numpy.array([x, y])
            superpixel_points.setdefault(superpixel, []).append(point)

    return {
        superpixel: _calculate_centroid(points)
        for superpixel, points in superpixel_points.items()
    }
