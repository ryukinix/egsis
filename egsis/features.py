from typing import List, Optional, Literal, Callable

import numpy as np
from skimage.feature import graycomatrix, graycoprops
from egsis.gabor import feature_extraction_gabor

FeaturesMethods = Literal["comatrix", "gabor"]


def _feature_extraction_comatrix_channel(channel: np.ndarray) -> np.ndarray:
    glcm = graycomatrix(
        channel,
        distances=[10],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=256,
        symmetric=True,
        normed=True
    )
    metrics = np.array([
        graycoprops(glcm, "correlation"),
        graycoprops(glcm, "energy"),
        graycoprops(glcm, "ASM"),
        graycoprops(glcm, "dissimilarity"),
    ])
    stat_functions: List[Callable[[np.ndarray], np.ndarray]] = [
        np.mean,
        np.median,
        np.var,
        np.std,
        lambda v: np.quantile(v, q=0.25)
    ]
    stats = [
        f(m)
        for m in metrics
        for f in stat_functions
    ]
    features = np.array(stats)
    return features


def feature_extraction_comatrix(img: np.ndarray) -> np.ndarray:
    """Img should be a rgb with 3 dimensions"""
    r = _feature_extraction_comatrix_channel(img[:, :, 0])
    g = _feature_extraction_comatrix_channel(img[:, :, 1])
    b = _feature_extraction_comatrix_channel(img[:, :, 2])
    return np.concatenate((r, g, b), axis=0)


def get_segment_by_label(
    img: np.ndarray,
    segments: np.ndarray,
    label: int,
    erase_color: Optional[int] = 0
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
    """Crop image as square window with max_radius and centroid point"""
    # FIXME(@lerax): qua 29 mar 2023 12:02:30:
    # Don't allow that values to be negative:
    #
    # y - max_radius
    # x - max_radius
    #
    # if they are, set it as 0.
    x, y = centroid

    start_heigth = y - max_radius
    end_heigth = y + max_radius
    start_width = x - max_radius
    end_width = x + max_radius

    if start_heigth < 0:
        end_heigth += abs(start_heigth)
        start_heigth = 0

    if start_width < 0:
        end_width += abs(start_width)
        start_width = 0

    s = img[
        start_heigth: end_heigth,
        start_width: end_width
    ]
    return s


def get_segment_by_label_cropped(
    img: np.ndarray,
    segments: np.ndarray,
    label: int,
    max_radius: int,
    centroid: List[int],
    erase_color: Optional[int] = 0
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


def euclidian_similarity_exp(u: np.ndarray, v: np.ndarray) -> np.floating:
    """Euclidian similarity exponential

    Using exponential 1/exp(v)
    """
    return np.exp(-euclidian_distance(u, v))


def manhattan_distance(u: np.ndarray, v: np.ndarray) -> np.floating:
    return sum(abs(p - q) for p, q in zip(u, v))


def manhattan_similarity_exp(u: np.ndarray, v: np.ndarray) -> np.floating:
    return np.exp(-manhattan_distance(u, v))


def manhattan_similarity_log(u: np.ndarray, v: np.ndarray) -> np.floating:
    return 1 / 1 + np.log(1 + manhattan_distance(u, v))


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
    label: int,
    max_radius: Optional[int] = None,
    centroid: Optional[List[int]] = None,
    erase_color: Optional[int] = None,
    feature_method: FeaturesMethods = "comatrix"
):
    """Return the features of the segment

    If centroid and max_radius are provided, it crops the image
    """
    feature_functions = {
        "comatrix": feature_extraction_comatrix,
        "gabor": feature_extraction_gabor
    }

    if any(param is not None for param in (max_radius, centroid)):
        image_segment = get_segment_by_label_cropped(
            img,
            segments,
            label,
            max_radius=max_radius,  # type: ignore
            centroid=centroid,  # type: ignore
            erase_color=erase_color
        )
    else:
        image_segment = get_segment_by_label(
            img,
            segments,
            label
        )
    # logger.debug(f"img: {image_segment}")
    return feature_functions[feature_method](image_segment)
