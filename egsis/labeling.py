"""
Module responsible to handle label creation of superpixels
"""

from typing import Tuple, Dict
import numpy


def create_label_matrix(shape=Tuple[int, int]) -> numpy.ndarray:
    """Create a label matrix without any labels, all values are zero.

    Zero means unlabelled in this context. The labels are non-zero
    integers.
    """
    return numpy.zeros(shape=shape, dtype=numpy.int64)


def set_superpixel_label(
    label_matrix: numpy.ndarray,
    superpixels: numpy.ndarray,
    superpixel: int,
    label: int
) -> numpy.ndarray:
    """Set the label of a specific superpixel"""
    label_matrix[superpixels == superpixel] = label
    return label_matrix


def get_superpixel_label(
    label_matrix: numpy.ndarray,
    superpixels: numpy.ndarray,
    superpixel: int,
):
    """Get the most ocurrent label in superpixel, ignoring unlabeled"""
    y = label_matrix[(label_matrix != 0) & (superpixels == superpixel)]
    histogram = numpy.bincount(y.flatten())
    most_occurent_label_at_superpixel = numpy.argmax(histogram) if histogram.size > 0 else 0

    return most_occurent_label_at_superpixel


def create_segmentation_mask(
    superpixels: numpy.ndarray,
    superpixels_by_label: Dict[int, int]
) -> numpy.ndarray:
    """Create segmentation mask as final output result"""
    label_matrix = create_label_matrix(superpixels.shape)
    for superpixel, label in superpixels_by_label.items():
        label_matrix = set_superpixel_label(
            label_matrix,
            superpixels,
            superpixel,
            label
        )

    return label_matrix
