from typing import List
from functools import lru_cache

import networkx as nx
import numpy as np

from loguru import logger


class LabeledComponentUnfolding:

    """
    Collective Dynamic Labeled Component Unfolding


    It can be used to solve Transductive Semi-Supervised problems.


    Parameters
    ----------
    n_classes : int
         Number of classes of a given problem. Use n_classes=2 for binary
         classification, for instance
    competition_level : float
         (lambda) varies between 0 and 1.
          0 means random walk and 1 max competition (recommended).

    max_iter : int
         The time limit parameter controls when the simulation should
         stop; it must be at least as large as the diameter of the network.


    Reference
    ---------
    Filipe Alves Neto Verri, Paulo Roberto Urio, Liang Zhao: “Network
    Unfolding Map by Edge Dynamics Modeling”, 2016, IEEE Transactions
    on Neural Networks and Learning Systems, vol. 29, no. 2,
    pp. 405-418, Feb. 2018. doi: 10.1109/TNNLS.2016.2626341;
    arXiv:1603.01182. DOI: 10.1109/TNNLS.2016.2626341.
    """

    def __init__(
        self,
        n_classes: int,
        competition_level: float = 1,
        scale_particles: float = 100,
        max_iter: int = 500,
    ):
        assert 0 <= competition_level <= 1, "Competition lvl should be: (0, 1]"
        assert n_classes >= 2, "Minimum classes are 2."
        self.competition_level = competition_level
        self.max_iter = max_iter
        self.n_classes = n_classes
        self.scale_particles = scale_particles
        self.classes = list(range(1, n_classes + 1))
        self.iterations = 0
        self.G: nx.Graph
        self.P: nx.Graph
        self.n: np.ndarray
        self.N: np.ndarray
        self.delta: np.ndarray
        hyperparams = ", ".join([
            f"{k}={v}" for k, v in {
                "n_classes": n_classes,
                "competition_level": competition_level,
                "scale_particles": scale_particles,
                "max_iter": max_iter,
            }.items()
        ])
        logger.info(f"hyperparams: {hyperparams}")

    def fit_predict(self, G: nx.Graph) -> List[nx.Graph]:
        """Fit complex network and predict new unlabeled data points.

        Each vertex of the complex_network should contains the
        label (if it's labeled) and it's features.

        Notes
        -----
        About the underlying Graph data structure and metadata:

        The label are stored at attribute 'label' at each node.
        Features are stored at attribute 'features' at each node.
        Weight of edges are stored at attribute 'weight' at each edge.

        Returns
        -------
        List of subgraphs containing the prediction for each class.
        They are ordered by class, which means: c=1 -> List[0]
        """
        self.init(G)
        for _ in range(self.max_iter):
            self.step(G)
            self.iterations += 1
        logger.trace(f"delta[0] = \n{self.delta[0]}")
        logger.trace(f"delta[1] = \n{self.delta[1]}")
        return self.unfold(G)

    def init(self, G: nx.Graph):
        """Initialize the LCU algorithm"""
        self.G = G
        self.n = self.n0(G)
        self.N = self.N0(G)
        self.delta = self.delta0(G)
        logger.debug(f"Initialized: G=(nodes={self.nodes},edges={self.edges})")

    def step(self, G: nx.Graph):
        """Execute iteration of LCU algorithm"""
        if (self.iterations+1) % 10 == 0:
            logger.trace(f"== iteration={self.iterations+1}/{self.max_iter}")

        for c in range(self.n_classes):
            self.P = self.probability(G)
            g_c = self.g(G, c)
            # NOTE: originally that product was nc x Pc, but I inverted due
            # matrix rules
            nc_product_pc = self.P[c] @ self.n[c]
            self.N[c] = np.diag(self.n[c]) @ self.P[c]
            self.n[c] = nc_product_pc + g_c
            self.delta[c] = self.delta[c] + self.N[c]

    def unfold(self, G: nx.Graph) -> List[nx.Graph]:
        return [
            self.subnetwork_of_class(G, c)
            for c in range(self.n_classes)
        ]

    def subnetwork_of_class(self, G: nx.Graph, c: int) -> nx.Graph:
        edges_subnetwork = []
        for (i, j) in G.edges:
            d = self.delta[:, i, j] + self.delta[:, j, i]
            if sum(d) == 0:  # unlabeled
                continue
            if np.argmax(d, axis=0) == c:
                edges_subnetwork.append((i, j))

        logger.debug(edges_subnetwork)
        logger.debug(f"class={c+1} subnetwork: edges={len(edges_subnetwork)}")
        sub_graph = nx.Graph()
        sub_graph.add_nodes_from(G.nodes)
        sub_graph.add_edges_from(edges_subnetwork)

        return sub_graph

    @property
    def nodes(self):
        return self.G.number_of_nodes()

    @property
    def edges(self):
        return self.G.number_of_edges()

    def p(self, G: nx.graph, i: int, j: int, c: int) -> float:
        """Individual probability of walk and survival of specific node

        Simulate particle survival + walk probability of class c going
        from v_i to v_j.
        """
        if label := G.nodes[j].get("label", 0):
            if label != c + 1:
                return 0

        edge_weight = G.edges[i, j]["weight"]
        walk = edge_weight / G.degree[i]
        survival = 1 - self.competition_level * self.sigma(G, i, j, c)
        # FIXME: weirdly, sigma function only outputs 1, 0.5  or 0
        # it doesn't makes sense
        # logger.trace(f"survival factor = {survival}")
        return walk * survival

    def probability(self, G: nx.Graph) -> np.ndarray:
        """Matrix with probabilities of particle surviving"""
        P = np.zeros(shape=self.N.shape)
        C, nodes, _ = P.shape
        for c in range(C):
            for i, j in G.edges:
                P[c, i, j] = self.p(G, i, j, c)

        return P

    def probability_of_new_particles(self, G: nx.Graph, c: int) -> np.ndarray:
        # FIXME: review this code based on the paper
        # equation on page 4
        node_degrees = [
            G.degree[node] for node in G.nodes
        ]
        total_degree_sum = sum(node_degrees)
        p = np.array([
            degree / total_degree_sum if G.nodes[node]["label"] == c + 1 else 0
            for node, degree in zip(G.nodes, node_degrees)
        ])
        return p

    @lru_cache
    def n0(self, G: nx.Graph) -> np.ndarray:
        """Initial active particles by vertex n_i[c]"""
        nodes = self.nodes
        edges = self.edges
        population = np.array([G.degree[v] / (2 * edges) for v in G.nodes])
        labels = np.zeros(shape=(self.n_classes, nodes))
        logger.debug(f"n0: classes={self.n_classes}, shape={labels.shape}")
        for idx, node in enumerate(G.nodes):
            label = G.nodes[node].get("label", 0)
            cls = label - 1
            if label != 0:
                labels[cls, idx] = label
        particles = labels * population * self.scale_particles
        logger.debug(f"n0: \n{particles}")
        return particles

    def N0(self, G: nx.Graph):
        """Initial N with current directed domination n_ij[c]
        """
        nodes = self.nodes
        return np.zeros(shape=(self.n_classes, nodes, nodes))

    def delta0(self, G: nx.Graph):
        """Row with cumulative domination delta_ij[c]

        Number of particles with label=c which moved from v_i to
        v_j until now.
        """
        nodes = self.nodes
        return np.zeros(shape=(self.n_classes, nodes, nodes))

    def sigma(self, G: nx.Graph, i: int, j: int, c: int) -> float:
        """Current relative subordination sigma_ij[c]

        The fraction of particles that do not belong to class c and
        have sucessfuly passed thorugh edge (i, j) in any direction at
        the current time.
        """
        S = np.sum((self.N[:, i, j] + self.N[:, j, i]).flatten())
        result: float
        if S > 0:
            result = 1 - ((self.N[c][i][j] + self.N[c][j][i]) / S)
        else:
            result = 1 - (1 / self.n_classes)

        return result

    def g(self, G: nx.Graph, c: int) -> np.ndarray:
        """Auxiliar function to compute new particles"""
        n0_sum = np.sum(self.n0(G)[c].flatten())
        n_sum = np.sum(self.n[c].flatten())
        p = self.probability_of_new_particles(G, c)
        g = p * max(0, n0_sum - n_sum)
        return g

    def classify_vertexes(self, sub_networks: List[nx.Graph]) -> nx.Graph:
        """Classify the unlabeled vertexes based on the edge domination"""
        G_pred = self.G.copy()
        neighbors_classes = np.zeros(shape=(self.n_classes, self.nodes))
        for c in range(self.n_classes):
            subnet = sub_networks[c]
            for j in subnet.nodes:
                neighbors_classes[c, j] = len(list(subnet.neighbors(j)))
        for node in G_pred.nodes:
            label = np.argmax(neighbors_classes[:, node], axis=0)
            G_pred.nodes[node]["label"] = label + 1

        return G_pred


# alias
LCU = LabeledComponentUnfolding
