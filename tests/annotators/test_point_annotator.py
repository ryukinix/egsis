from egsis.annotators import PointAnnotator
import ipywidgets as widgets
from skimage.data import retina


def test_point_annotator_initialization():
    annotator = PointAnnotator(
        image=retina(),
        classes=["retina", "fg"]
    )
    assert isinstance(annotator.display(), widgets.Widget)
