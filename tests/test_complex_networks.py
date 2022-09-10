import numpy
import networkx as nx

from egsis import complex_networks


def test_complex_network_from_segments():
    example = numpy.array(
        [
            [1, 2, 2],
            [1, 2, 2],
            [0, 0, 3],
        ]
    )
    expected = nx.Graph({
        0: {1, 2, 3},
        1: {0, 2},
        2: [0, 1, 3],
        3: [0, 2],
    })

    got = complex_networks.complex_network_from_segments(example)
    assert nx.utils.graphs_equal(got, expected), f"{expected}, {got}"
