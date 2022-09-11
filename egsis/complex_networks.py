from typing import Dict, Set

import numpy
import networkx

from egsis import superpixels


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

    height, width = segments.shape
    graph: Dict[int, Set[int]] = dict()
    for x in range(height):
        for y in range(width):
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


def draw_complex_network(
    graph: networkx.Graph,
    segments: numpy.ndarray,
    ax=None
):
    """Draw complex network in predefined layout based in the segments.

    This function draws the vertexes in the centroid of its segment.

    It can be used to draw complex_network semantically over the image
    and its segments. Useful for comprehension.
    """
    networkx.draw_networkx(
        graph,
        node_size=10,
        arrows=False,
        alpha=1,
        ax=ax,
        node_color='#ff0000',
        edge_color='#f8fc03',
        width=0.8,
        with_labels=False,
        pos=superpixels.superpixel_centroids(segments)
    )
