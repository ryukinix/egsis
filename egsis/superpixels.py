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

    Returns
    ------
    centroids: Dict[int, List[int]]
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


def _check_point_inside_superpixel(
    segments: numpy.ndarray,
    superpixel_label: int,
    point: Tuple[int, int],
):
    height, width = segments.shape
    y, x = point
    restrictions = [
        x < width,
        y < height,
        x > 0,
        y > 0
    ]

    if not all(restrictions):
        return False

    return segments[point] == superpixel_label


def superpixels_max_radius(
    segments: numpy.ndarray,
    centroids: Dict[int, List[int]]
) -> int:
    """Calculate the max diameter size in the matrix of segments.

    By using a squared expantion, it tries to calculate optimally the
    max diameter of all superpixels.

    Parameters
    ----------
    segments : numpy.ndarray shape=(height, width)
         Matrix of segments, each value being the label of superpixel
    centroids : Dict[int, List[int]]
         Map of centroids for each superpixel.
         Each key represents the label of superpixel
         Values represents the point of the centroid.

    Return
    ------
    max_diameter : int
        To be used to crop superpixels
    """
    radius = 1
    superpixel_labels = set(centroids.keys())
    while len(superpixel_labels) > 1:
        for superpixel in superpixel_labels.copy():
            centroid_pos = centroids[superpixel]
            y, x = centroid_pos
            search_points = [
                (x, y + radius),
                (x + radius, y),
                (x, y - radius),
                (x - radius, y),
                (x + radius, y + radius),
                (x + radius, y - radius),
                (x - radius, y + radius),
                (x - radius, y - radius),
            ]
            square_inside_of_superpixel = any(
                _check_point_inside_superpixel(
                    segments,
                    superpixel,
                    p
                )
                for p in search_points
            )
            if not square_inside_of_superpixel:
                superpixel_labels.remove(superpixel)
        radius += 1
    return radius
