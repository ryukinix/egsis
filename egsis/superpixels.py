from typing import Dict, List, Tuple

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


def superpixels_max_window_size(
    segments: numpy.ndarray,
    centroids: Dict[int, List[int]]
) -> Tuple[str, str]:
    """Calculate the max window size present in the matrix of segments.

    By using a radial expantion, it tries to calculate optimally the
    max window size for all superpixels.

    Parameters
    ----------
    segments : numpy.ndarray (height, width)
         Matrix of segments with each value being the label of superpixels
    centroids : Dict[int, List[int]]
         Map of centroids for each superpixel.
         Each key represents the label of superpixel
         Values represents the point of the centroid.

    Return
    ------
    (max_height, max_width) to crop superpixels
    """
