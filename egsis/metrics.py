import numpy as np


def _check_if_they_are_binarized(y_true: np.ndarray, y_pred: np.ndarray):
    set_of_binary = {0, 1}
    set_of_values = np.unique(np.concatenate([y_true, y_pred]))
    if any(value not in set_of_binary for value in set_of_values):
        raise ValueError("such metric cannot be used in non-binary labels")


def pixel_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    equal_pixels = (y_true == y_pred).sum()
    total = y_true.size
    acc = equal_pixels / total
    return acc


def iou(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    _check_if_they_are_binarized(y_true, y_pred)
    intersection = y_true & y_pred
    union = y_true | y_pred
    return intersection.sum() / union.sum()


def f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    _check_if_they_are_binarized(y_true, y_pred)
    intersection = y_true & y_pred
    total_size = y_true.sum() + y_pred.sum()
    return (2 * intersection.sum()) / total_size


def err(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """This function calculates the error rate, also known as the
     mis-segmentation rate, between the true and predicted values.

    Parameters
    ---------
    y_true : np.ndarray
        The ground truth binary labels. The binary label indicates
        whether each pixel is within the region of interest (ROI) or
        not.
    y_pred : np.ndarray:
        The predicted binary labels. The binary label indicates
        whether each pixel is predicted to be within the ROI or not.

    Returns
    -------
    err : float

    The mis-segmentation rate. This is calculated as the
    number of mis-segmented pixels (pixels that are in the union of
    the true and predicted labels but not in the true labels) divided
    by the total number of pixels in the ROI.
    """
    roi_pixels = y_true
    mis_segmentation = (y_true | y_pred) - y_true
    return mis_segmentation.sum() / roi_pixels.sum()
