from typing import Dict, Set

import numpy
import networkx
from skimage.segmentation import find_boundaries


def complex_network_from_segments(segments: numpy.ndarray) -> networkx.Graph:
    """Build complex network from segments of superpixel.

    Parameters:
    -----------
    segments: numpy.array
        Segments of image as superpixel

    Returns
    -------
    Undirected Graph (networkx.Graph)
    """

    boundaries = find_boundaries(segments)
    height, width = segments.shape
    graph: Dict[int, Set[int]] = dict()
    for x in range(height):
        for y in range(width):
            if boundaries[x][y]:
                vertex = segments[x][y]
                neighbors = set()
                if x > 0:
                    up = segments[x-1][y]
                    if up != vertex:
                        neighbors.add(up)
                if x + 1 < height:
                    down = segments[x+1][y]
                    if down != vertex:
                        neighbors.add(down)
                if y > 0:
                    left = segments[x][y-1]
                    if left != vertex:
                        neighbors.add(left)
                if y + 1 < width:
                    right = segments[x][y+1]
                    if right != vertex:
                        neighbors.add(right)
                graph.setdefault(vertex, set())
                graph[vertex] |= neighbors
    return networkx.Graph(graph)
