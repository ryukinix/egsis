from typing import Optional, Protocol
import networkx as nx
from egsis.complex_networks import complex_network_from_segments
from sklearn.neighbors import NearestNeighbors
import numpy as np


class GraphBuilder(Protocol):
    def build(self, segments: np.ndarray, features: np.ndarray) -> nx.Graph:
        ...


class GraphBuilderPlain:
    def build(self, segments: np.ndarray, features: Optional[np.ndarray] = None) -> nx.Graph:
        return complex_network_from_segments(segments)


class GraphBuilderKNN:
    def __init__(self, k: int = 3, sensitivity: float = 10.0):
        self.k = k
        self.sensitivity = sensitivity

    def build(self, segments: np.ndarray, features: np.ndarray) -> nx.Graph:
        G = complex_network_from_segments(segments)
        nodes = list(G.nodes)
        k = min(self.k, len(nodes) - 1) if len(nodes) > 1 else 0
        if k > 0:
            nn = NearestNeighbors(n_neighbors=k)
            nn.fit(features)
            dists, indices = nn.kneighbors(features)
            for i in range(len(nodes)):
                for j, neighbor_idx in enumerate(indices[i]):
                    if i != neighbor_idx and not G.has_edge(nodes[i], nodes[neighbor_idx]):
                        weight = np.exp(-dists[i][j] / self.sensitivity)
                        G.add_edge(nodes[i], nodes[neighbor_idx], weight=weight)
        return G
