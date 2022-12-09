import networkx as nx
import numpy

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
        2: {0, 1, 3},
        3: {0, 2},
    })

    got = complex_networks.complex_network_from_segments(example)
    assert nx.utils.graphs_equal(got, expected), f"{expected}, {got}"


def test_draw_complex_network():
    complex_networks.draw_complex_network(
        nx.Graph(
            {
                0: {1},
                1: {0},
            },
        ),
        numpy.array(
            [
                [0, 0],
                [1, 1]
            ]
        )
    )
