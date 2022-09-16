from typing import List

import numpy as np


def feature_extraction(img: np.ndarray) -> np.ndarray:
    """Multidimensional fourier transform

    NOTE(@lerax): dom 11 set 2022 09:49:35
    Maybe I should use PCA to reduce the high dimensional space.
    """
    return np.fft.fftshift(np.fft.fftn(img))


def get_segment_by_label(
    img: np.ndarray,
    segments: np.ndarray,
    label: int,
    erase_color: int = 0
) -> np.ndarray:
    """Return only the segment selected"""
    img_copy = img.copy()
    if erase_color is not None:
        img_copy[segments != label] = erase_color
    return img_copy


def crop_image(
    img: np.ndarray,
    max_radius: int,
    centroid: List[int],
):
    x, y = centroid
    return img[
        y - max_radius: y + max_radius,
        x - max_radius: x + max_radius
    ]


def get_segment_by_label_cropped(
    img: np.ndarray,
    segments: np.ndarray,
    label: int,
    max_radius: int,
    centroid: List[int],
    erase_color: int = 0
):
    segment = get_segment_by_label(
        img,
        segments,
        label,
        erase_color=erase_color
    )
    return crop_image(segment, max_radius, centroid)


def euclidian_distance(u: np.ndarray, v: np.ndarray) -> np.floating:
    """Compute euclidian distance between two vectors"""
    return np.linalg.norm(u - v)


def euclidian_similarity(u: np.ndarray, v: np.ndarray) -> np.floating:
    """Euclidian similarity function

    Can be used in multiple dimensional arrays.
    """
    return 1 / (1 + euclidian_distance(u, v))


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> np.floating:
    """Cosine similiarty function

    Can only be used in 1D arrays.
    """
    u_shape, v_shape = u.shape, v.shape
    if u_shape != v_shape:
        msg = f"Vectors with different shapes: u={u_shape}; v={v_shape}"
        raise ValueError(msg)
    if len(u_shape) > 1 or len(v_shape) > 1:
        u = u.flatten()
        v = v.flatten()

    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def feature_extraction_segment(
    img: np.ndarray,
    segments: np.ndarray,
    label: int
):
    """Return the features of the segment"""
    return feature_extraction(
        get_segment_by_label(
            img,
            segments,
            label
        )
    )
