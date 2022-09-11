import numpy as np


def feature_extraction(img: np.ndarray) -> np.ndarray:
    """For a while, multidimensional fft"""
    return np.fft.fftshift(np.fft.fftn(img))


def get_segment_using_label(
    img: np.ndarray,
    segments: np.ndarray,
    label: int
) -> np.ndarray:
    """Return only the segment selected"""
    img_copy = img.copy()
    img_copy[segments == label] = 0
    return img_copy


def euclidian_distance(u, v):
    """Compute euclidian distance between two vectors"""
    return np.linalg.norm(u - v)


def feature_extraction_segment(
    img: np.ndarray,
    segments: np.ndarray,
    label: int
):
    """Return the features of the segment"""
    return feature_extraction(
        get_segment_using_label(
            img,
            segments,
            label
        )
    )
