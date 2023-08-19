"""
Module responsible to handle label creation of superpixels
"""

from typing import Tuple
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
    superpixel_mask = superpixels[superpixels == superpixel]
    label_matrix[superpixel_mask] = label
    return label_matrix


def get_superpixel_label(
    label_matrix: numpy.ndarray,
    superpixels: numpy.ndarray,
    superpixel: int,
):
    """Get the most ocurrent label in superpixel"""
    histogram = numpy.bincount(label_matrix[superpixels == superpixel].flatten())
    most_occurent_label_at_superpixel = numpy.argmax(histogram)

    return most_occurent_label_at_superpixel
