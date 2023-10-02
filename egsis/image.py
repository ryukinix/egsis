"""
Module focused in:

- image visualization
- blending images (e.g.: alpha-blending)
- annotation of images
"""
from typing import Dict

from PIL import ImageColor
from skimage.segmentation import mark_boundaries
import numpy as np


def alpha_blend(img1: np.ndarray, img2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend two images based in alpha coefficient between [0, 1]

    Parameters
    ----------

    img1 : np.ndarray
        Main layer image
    img2 : np.ndarray
        Secondary image layer
    alpha : float
        Defaults to 0.5
        Maximum value 1 means total opacity of img2
        Minimum value 0 means total opacity of img1
    """
    blend = img1 * (1 - alpha) + img2 * alpha
    return blend.astype(np.uint8)


def segmentation_mask_blend(img: np.ndarray, mask: np.ndarray, labels_by_color: Dict[int, str]):
    """Generates colorful segmentation mask over main image with boundaries marked

    Parameters
    ----------

    img : np.ndarray
        Main layer image
    mask : np.ndarray
        Matrix with segmentation labels for each pixel
    labels_by_color: Dict[int, str]
        keys are classes of segmentation and values are color in hex code format

    """
    img_mask = img.copy()
    for label, hex_color in labels_by_color.items():
        img_mask[mask == label] = ImageColor.getcolor(hex_color, "RGB")
    blend = alpha_blend(img, img_mask)
    return mark_boundaries(blend, mask)
