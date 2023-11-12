"""
Gabor filter feature extraction method
"""

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import gabor_kernel

from functools import lru_cache


@lru_cache()
def _generate_kernels():
    # prepare filter bank kernels
    kernels = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for frequency in (0.05, 0.25):
                kernel = np.real(
                    gabor_kernel(
                        frequency,
                        theta=theta,
                        sigma_x=sigma,
                        sigma_y=sigma
                    )
                )
                kernels.append(kernel)
    return kernels


def _compute_features(image, kernels):
    feats = np.zeros((len(kernels), 5), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
        feats[k, 2] = filtered.std()
        feats[k, 3] = np.quantile(filtered, q=0.25)
        feats[k, 4] = np.median(filtered)

    return feats.flatten()


def feature_extraction_gabor(image: np.ndarray) -> np.ndarray:
    kernels = _generate_kernels()
    r = _compute_features(image[:, :, 0], kernels)
    g = _compute_features(image[:, :, 1], kernels)
    b = _compute_features(image[:, :, 2], kernels)
    return np.concatenate([r, g, b], axis=0)
