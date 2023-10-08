from egsis.annotators import PointAnnotator
from ipycanvas import Canvas
from skimage.data import retina


def test_point_annotator_initialization():
    assert isinstance(PointAnnotator(image=retina()).display(), Canvas)
