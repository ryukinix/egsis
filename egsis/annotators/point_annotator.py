"""
The point annotator is a abstraction of selecting arbitrary points using mouse
"""

from typing import List, Tuple
from dataclasses import dataclass


from ipycanvas import Canvas
from ipywidgets import Output
from skimage.transform import resize
from skimage.util import img_as_ubyte
import numpy as np


@dataclass
class LabeledPoint:
    label: str
    color: np.ndarray  # FIXME: maybe hexcode later
    point: Tuple[int, int]


class PointAnnotator:

    def __init__(self, image: np.ndarray, resize_f: float = 0.5):
        assert resize_f > 0
        self.image = img_as_ubyte(image)
        self.resize_f = resize_f
        width, height, _ = image.shape
        self.width_resized = width * resize_f
        self.height_resized = height * resize_f
        self.canvas = Canvas(
            width=self.width_resized,
            height=self.height_resized
        )
        self.output = Output()
        self.labels = [1, 2]  # FIXME: this should be dynamic
        self.colors = np.array(
            [
                [[228, 183, 164]],
                [[183, 228, 164]]
            ]
        )
        self.colors_outline = self.colors * 1.1
        self.data_points: List[LabeledPoint] = []
        self._initialize_image()
        self._initialize_mouse()

    def _get_original_cordinate_system(self, x, y) -> Tuple[int, int]:
        scale_up = 1 / self.resize_f
        return round(x * scale_up), round(y * scale_up)

    def _initialize_image(self):
        resized_shape = (self.width_resized, self.height_resized)
        img_resized = resize(self.image, resized_shape)
        self.canvas.put_image_data(img_as_ubyte(img_resized), 0, 0)

    def _initialize_mouse(self):
        @self.output.capture()
        def handle_click(x, y):
            label = 1
            self.data_points.append(
                LabeledPoint(
                    label=str(label),
                    color=self.colors[label],
                    point=self._get_original_cordinate_system(x, y)
                )
            )
            self.draw_circle(x, y, label=label)
        self.canvas.on_mouse_up(handle_click)

    @property
    def data(self) -> List[LabeledPoint]:
        """Data points labeled as list of LabeledPoint.
        """
        return self.data_points

    def display(self):
        return self.canvas

    def draw_circle(self, x, y, label: int):
        alpha = [0.5]
        r = [5]
        color_fill = np.array([self.colors[label]])
        color_outline = np.array([self.colors_outline[label]])
        self.canvas.fill_styled_circles(x, y, r, color=color_fill, alpha=alpha)
        self.canvas.line_width = 2
        self.canvas.stroke_styled_circles(x, y, r, color=color_outline)
