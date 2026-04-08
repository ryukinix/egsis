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
        # Implementação atual (baseada apenas em adjacência de grade)
        return complex_network_from_segments(segments)

class GraphBuilderKNN:
    def __init__(self, k: int = 3, sensitivity: float = 10.0):
        self.k = k
        self.sensitivity = sensitivity

    def build(self, segments: np.ndarray, features: np.ndarray) -> nx.Graph:
        # 1. Base espacial
        G = complex_network_from_segments(segments)
        # 2. KNN semântico
        nodes = list(G.nodes)
        if len(nodes) <= self.k:
            k = len(nodes) - 1
        else:
            k = self.k
            
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(features)
        dists, indices = nn.kneighbors(features)
        
        for i in range(len(nodes)):
            for j, neighbor_idx in enumerate(indices[i]):
                node_i = nodes[i]
                node_j = nodes[neighbor_idx]
                if node_i != node_j and not G.has_edge(node_i, node_j):
                    weight = np.exp(-dists[i][j] / self.sensitivity)
                    G.add_edge(node_i, node_j, weight=weight)
        return G
