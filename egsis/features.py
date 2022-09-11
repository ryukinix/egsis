import numpy as np


def feature_extraction(img: np.ndarray) -> np.ndarray:
    """Real part of multidimensional fourier transform

    Using flatten to make it possible use cosine-similarity.

    NOTE(@lerax): dom 11 set 2022 09:49:35
    Maybe I should use PCA to reduce the high dimensional space.
    """
    return abs(np.fft.fftshift(np.fft.fftn(img)).flatten())


def get_segment_using_label(
    img: np.ndarray,
    segments: np.ndarray,
    label: int
) -> np.ndarray:
    """Return only the segment selected"""
    img_copy = img.copy()
    img_copy[segments == label] = 0
    return img_copy


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
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


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
