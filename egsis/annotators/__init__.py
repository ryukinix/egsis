"""
Package for widget of point-based segmentation annotation to help create initial labels
for algorithm EGSIS.

It combines two libraries:

- ipycanvas for main logic of annotation
- ipywidgets for box/dropdown menus, color select, class selection, buttons
"""

from egsis.annotators.point_annotator import PointAnnotator


__all__ = [
    "PointAnnotator"
]
