from typing import Dict, Set, Callable, Tuple

import numpy
import networkx

from egsis import superpixels
from egsis import features
from egsis import labeling


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


def compute_node_labels(
    graph: networkx.Graph,
    segments: numpy.ndarray,
    labels: numpy.ndarray
) -> networkx.Graph:
    """Add node label based on superpixel and img pixel labels"""
    for v in graph.nodes:
        graph.nodes[v]["label"] = labeling.get_superpixel_label(
            label_matrix=labels,
            superpixels=segments,
            superpixel=v
        )

    return graph


def compute_node_features(
    graph: networkx.Graph,
    img: numpy.ndarray,
    segments: numpy.ndarray,
    feature_method: features.FeaturesMethods
) -> networkx.Graph:
    centroids = superpixels.superpixel_centroids(segments)
    max_radius = superpixels.superpixels_max_radius(segments, centroids)
    for v in graph.nodes:
        graph.nodes[v]["features"] = features.feature_extraction_segment(
            img=img,
            segments=segments,
            label=v,
            max_radius=max_radius,
            centroid=centroids[v],
            feature_method=feature_method
        )
    return graph


def compute_edge_weights(
    graph: networkx.Graph,
    similarity_function: Callable
) -> networkx.Graph:
    """It assumes that the graph already have node with features."""
    similarities: Dict[Tuple[int, int], float] = {}
    for (vi, vj) in graph.edges:
        xi, xj = graph.nodes[vi]["features"], graph.nodes[vj]["features"]
        similarities[(vi, vj)] = similarity_function(xi, xj)
    networkx.set_edge_attributes(graph, similarities, "weight")
    return graph


def draw_complex_network(
    graph: networkx.Graph,
    segments: numpy.ndarray,
    ax=None,
    node_color='#ff0000',
    node_size=10,
    color_map=None,
):
    """Draw complex network in predefined layout based in the segments.

    This function draws the vertexes in the centroid of its segment.

    It can be used to draw complex_network semantically over the image
    and its segments. Useful for comprehension.
    """
    networkx.draw_networkx(
        graph,
        node_size=node_size,
        arrows=False,
        alpha=1,
        ax=ax,
        node_color=node_color,
        edge_color='#f8fc03',
        width=0.8,
        with_labels=False,
        pos=superpixels.superpixel_centroids(segments)
    )
