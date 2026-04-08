import networkx as nx


def prepare_graph_for_lcu(G: nx.Graph) -> tuple[nx.Graph, dict, dict]:
    mapping = {node: i for i, node in enumerate(G.nodes)}
    rev_mapping = {i: node for node, i in mapping.items()}
    G_mapped = nx.relabel_nodes(G, mapping)
    return G_mapped, mapping, rev_mapping
