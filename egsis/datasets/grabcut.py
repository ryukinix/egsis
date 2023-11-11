"""
Module of interfacing and reading the data of grabcut dataset
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List
from os.path import dirname, join
import numpy as np
import os
import random

from egsis import image
import egsis

# get correct relative path datasets/grabcut
DATASET_BASEPATH = join(dirname(dirname(egsis.__file__)), "datasets/grabcut")


@dataclass
class LabeledImage:
    fname: str
    data: np.ndarray
    lasso: np.ndarray
    segmentation: np.ndarray  # ground truth
    rect: np.ndarray

    def y_train(self):
        y_train = self.lasso.copy()
        y_train[y_train == 64] = 1  # background
        y_train[y_train == 128] = 0  # 0 means unlabelled for egsis
        y_train[y_train == 255] = 2  # foreground
        return y_train

    def y_true(self):
        y_true = self.segmentation.copy()
        y_true[y_true == 128] = 1
        y_true[y_true == 255] = 1
        return y_true


def _list_fnames() -> List[str]:
    """Get image names with extension"""
    datadir = Path(DATASET_BASEPATH) / "data"
    images = os.listdir(datadir)
    return [
        fname
        for fname in images
    ]


@lru_cache()
def get_labeled_image_by_name(fname: str) -> LabeledImage:
    basedir = Path(DATASET_BASEPATH)
    datadir = basedir / "data"
    lassodir = basedir / "lasso"
    rectdir = basedir / "rect"
    segmentationdir = basedir / "segmentation"
    fname_without_extension, _ = fname.split(".")
    name_bmp = fname_without_extension + ".bmp"

    return LabeledImage(
        fname=fname,
        data=image.read_image_from_file(datadir / fname),
        segmentation=image.read_image_from_file(segmentationdir / name_bmp),
        lasso=image.read_image_from_file(lassodir / name_bmp),
        rect=image.read_image_from_file(rectdir / name_bmp)
    )


def get_random_image() -> LabeledImage:
    names = _list_fnames()
    name = random.choice(names)
    return get_labeled_image_by_name(name)


def get_all_dataset() -> List[LabeledImage]:
    images = _list_fnames()
    return [
        get_labeled_image_by_name(image_name)
        for image_name in images
    ]
