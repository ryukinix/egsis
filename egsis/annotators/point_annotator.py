"""
The point annotator is a abstraction of selecting arbitrary points using mouse
"""

from typing import List, Tuple, Dict
from dataclasses import dataclass
import math

from ipycanvas import Canvas
from skimage.transform import resize
from skimage.util import img_as_ubyte
import ipywidgets as widgets
import numpy as np

from egsis.image import get_color_from_hexcode
from egsis.labeling import create_label_matrix


@dataclass
class LabeledPoint:
    label: str
    color: str
    radius: float
    point: Tuple[int, int]


class PointAnnotator:

    def __init__(self, image: np.ndarray, classes: List[str], resize_f: float = 0.5):
        assert resize_f > 0, "resize_f should great than zero"
        assert len(classes) >= 2, "classes should contains at least two elements"
        self.image = img_as_ubyte(image)
        self.classes = classes
        self.resize_f = resize_f
        width, height, _ = image.shape
        self.width_resized = width * resize_f
        self.height_resized = height * resize_f
        self.canvas = Canvas(
            width=self.width_resized,
            height=self.height_resized
        )
        self.output = widgets.Output()
        self.data_points: List[LabeledPoint]
        self.app: widgets.Widget
        self._initialized_label_colors: Dict[int, str] = {}
        self._initialize_image()
        self._initialize_mouse()
        self._initialize_data_points()
        self._initialize_control_panel()

    def _initialize_data_points(self):
        self.data_points = []

    def _lock_color_when_label_is_initialized(self):
        if self.label.value in self._initialized_label_colors:
            self.color.value = self._initialized_label_colors[self.label.value]
            self.color.disabled = True
        else:
            self.color.disabled = False

    def _initialize_control_panel(self):
        self.label = widgets.Dropdown(
            options=[(cls, idx)for idx, cls in enumerate(self.classes)],
            value=0,
            description='Label: ',
        )
        self.color = widgets.ColorPicker(
            concise=False,
            description='Label color: ',
            value='#00ff00',
            disabled=False
        )

        def lock_color(change):
            self._lock_color_when_label_is_initialized()

        self.label.observe(lock_color, names='value')
        self.radius = widgets.FloatSlider(
            value=5,
            min=5,
            max=15.0,
            step=0.1,
            description='Radius:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.1f',
        )
        self.restart = widgets.Button(
            description='Restart',
            disabled=False,
            button_style='warning',
            tooltip='Restart the labelling process',
            icon='restart'
        )

        def _restart(button):
            self.canvas.clear()
            self.color.disabled = False
            self._initialized_label_colors = {}
            self._initialize_data_points()
            self._initialize_image()
        self.restart.on_click(_restart)

        self.control_panel = widgets.VBox(
            [
                self.radius,
                self.color,
                self.label,
                self.restart
            ]
        )
        self.app = widgets.HBox(
            [self.canvas, self.control_panel]
        )

    def _initialize_image(self):
        resized_shape = (self.width_resized, self.height_resized)
        img_resized = resize(self.image, resized_shape)
        self.canvas.put_image_data(img_as_ubyte(img_resized), 0, 0)

    def _initialize_mouse(self):
        @self.output.capture()
        def handle_click(x, y):
            label = self.label.value
            self.data_points.append(
                LabeledPoint(
                    label=self.label.label,
                    color=self.color.value,
                    # HACK(@lerax): seg 09 out 2023 23:00:15
                    # for god sake, the x y are being swap
                    # at some place!!! I'm swaping this again
                    # to fix the problem
                    point=self._get_original_cordinate_system(y, x),
                    radius=self.radius.value / self.resize_f
                )
            )
            if label not in self._initialized_label_colors:
                self._initialized_label_colors[label] = self.color.value
            self._lock_color_when_label_is_initialized()
            self._draw_circle(x, y, label=label)
        self.canvas.on_mouse_up(handle_click)

    def _get_original_cordinate_system(self, x, y) -> Tuple[int, int]:
        scale_up = 1 / self.resize_f
        return round(x * scale_up), round(y * scale_up)

    def _draw_circle(self, x, y, label: int):
        alpha = [0.5]
        r = [self.radius.value]
        color_fill = np.array([get_color_from_hexcode(self.color.value)])
        color_outline = 0.8 * color_fill
        self.canvas.fill_styled_circles(x, y, r, color=color_fill, alpha=alpha)
        self.canvas.line_width = 2
        self.canvas.stroke_styled_circles(x, y, r, color=color_outline)
        self.canvas.save()

    @property
    def data(self) -> List[LabeledPoint]:
        """Data points labeled as list of LabeledPoint.
        """
        return self.data_points

    @property
    def labels_by_color(self) -> Dict[int, str]:
        """Useful for knowing colors to create segmentation mask later"""
        return {
            label + 1: color  # label 0 means unlabbeled in this context
            for label, color in
            self._initialized_label_colors.items()
        }

    @property
    def labels_by_name(self) -> Dict[int, str]:
        """Mapping labels (numbers) with the class name"""
        return {
            label + 1: name
            for label, name in
            enumerate(self.classes)
        }

    def _get_label_by_name(self, label_name: str) -> int:
        return self.classes.index(label_name) + 1

    @property
    def label_matrix(self) -> np.ndarray:
        w, h, _ = self.image.shape
        label_matrix = create_label_matrix(shape=(w, h))
        for labeled_point in self.data:
            label = self._get_label_by_name(labeled_point.label)
            x, y = labeled_point.point
            radius = labeled_point.radius

            pixel_candidates = []
            for sqr_x in range(round(x - radius), round(x + radius)):
                for sqr_y in range(round(y - radius), round(y + radius)):
                    pixel_candidates.append((sqr_x, sqr_y))

            for pixel in pixel_candidates:
                circle_center = labeled_point.point
                if check_point_in_circle(circle_center, radius, pixel):
                    label_matrix[pixel] = label

        return label_matrix

    def display(self):
        return self.app


def check_point_in_circle(
    circle_center: Tuple[int, int],
    circle_radius: float,
    point: Tuple[int, int]
):
    c_x, c_y = circle_center
    x, y = point
    distance_between_point_circle = math.sqrt((x - c_x) ** 2 + (y - c_y) ** 2)
    return distance_between_point_circle <= circle_radius
