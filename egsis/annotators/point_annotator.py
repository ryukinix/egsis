"""
The point annotator is a abstraction of selecting arbitrary points using mouse
"""

from typing import List, Tuple
from dataclasses import dataclass


from ipycanvas import Canvas
import ipywidgets as widgets
from skimage.transform import resize
from skimage.util import img_as_ubyte
import numpy as np

from egsis.image import get_color_from_hexcode


@dataclass
class LabeledPoint:
    label: str
    color: np.ndarray  # FIXME: maybe hexcode later
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
        self._initialize_image()
        self._initialize_mouse()
        self._initialize_data_points()
        self._initialize_control_panel()

    def _initialize_data_points(self):
        self.data_points = []

    def _initialize_control_panel(self):
        self.label = widgets.Dropdown(
            options=[(cls, idx)for idx, cls in enumerate(self.classes)],
            value=0,
            description='Label: ',
        )
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
        self.color = widgets.ColorPicker(
            concise=False,
            description='Label color: ',
            value='#0000ff',
            disabled=False
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
                    point=self._get_original_cordinate_system(x, y)
                )
            )
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

    def display(self):
        return self.app
