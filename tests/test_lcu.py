import random
import pytest
import networkx as nx

from egsis import LCU


@pytest.fixture
def graph() -> nx.Graph:
    n_nodes = 10
    n_edges = 30
    r = lambda: random.randint(0, n_nodes)  # noqa
    nodes = [x for x in range(n_nodes)]
    initial_edges = [(r(), r()) for x in range(n_edges)]
    edges = [(u, v, {"weight": random.random()})
             for u, v in initial_edges if u != v]
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    for u in G.nodes:
        G.nodes[u]["label"] = random.randint(0, 2)
    return G


def test_lcu_sanity_check(graph):
    """Basic test checking sanity"""
    lcu = LCU(
        n_classes=2,
        competition_level=1,
        max_iter=10
    )
    sub_networks = lcu.fit_predict(graph)
    assert len(sub_networks) == 2
    g1, g2 = sub_networks
    # disjunct sets
    assert len(g1.edges) > 0
    assert len(g2.edges) > 0
    assert set(g1.edges) & set(g2.edges) == set()
    assert set(g1.edges) | set(g2.edges) == set(graph.edges)
