"""
Module of interfacing and reading the data of grabcut dataset
"""

from typing import List
from dataclasses import dataclass

import numpy as np


DATASET_BASEPATH = "datasets/grabcut"


@dataclass
class LabeledImage:
    name: str
    data: np.ndarray
    lasso: np.ndarray
    rect: np.ndarray
    segmentation: np.ndarray  # ground truth


def build_label_matrix(lasso_or_rect: np.ndarray) -> np.ndarray:
    pass


def get_random_image() -> LabeledImage:
    pass


def get_all_dataset() -> List[LabeledImage]:
    pass


def get_labeled_image_by_name(name: str) -> LabeledImage:
    pass
