from typing import List

import networkx as nx
import numpy as np


class LabeledComponentUnfolding:

    """
    Collective Dynamic Labeled Component Unfolding


    It can be used to solve Semi-Supervised problems.


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
        max_iter: int = 500,
        initial_population_size: int = 50,
    ):
        assert competition_level > 0 and competition_level <= 1
        self.competition_level = competition_level
        self.max_iter = max_iter
        self.n_classes = n_classes
        self.initial_population_size = initial_population_size
        self.classes = list(range(1, n_classes + 1))
        self.iterations = 0
        self.G: nx.Graph
        self.P: nx.Graph
        self.n: np.ndarray
        self.N: np.ndarray
        self.delta: np.ndarray

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
        return self.unfold(G)

    @property
    def nodes(self):
        return self.G.number_of_nodes()

    def n0(self, G: nx.Graph):
        """Initial active particles by vertex n_i[c]"""
        nodes = self.nodes
        particles = np.ones(shape=(self.n_classes, nodes))
        return particles * self.initial_population_size

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
        """Matrix with current relative domination sigma_ij[c]

        Number of particles with label=c which moved from v_i to
        v_j in the current time.
        """
        S = np.sum((self.N[c] + self.N[c].T).flatten())
        result: float
        if S <= 0:
            result = 1 - (1 / self.n_classes)
        else:
            result = 1 - (self.N[c][i][j] + self.N[c][j][i]) / S

        return result

    def p(self, G: nx.graph, i: int, j: int, c: int) -> float:
        """Individual probability of walk and survival of specific node

        Simulate particle survival + walk probability of class c going
        from v_i to v_j.
        """
        # FIXME: set the case when labeled node with different class c
        # result should be p=0 (sink case, absorption)

        edge_weight = G.edges[i, j]["weight"]
        total = sum(G.edges[i, x]["weight"] for x in G.neighbors(i))
        walk = edge_weight / total
        survival = 1 - self.competition_level * self.sigma(G, i, j, c)

        return walk * survival

    def probability(self, G: nx.Graph) -> np.ndarray:
        """Matrix with probabilities of particle survival"""
        P = np.zeros(shape=self.N.shape)
        C, nodes, _ = P.shape
        for c in range(C):
            for i, j in G.edges:
                P[c, i, j] = self.p(G, i, j, c)

        return P

    def probability_of_new_particles(self, G: nx.Graph, c: int) -> np.ndarray:
        node_degrees = [
            G.degree[node] for node in G.nodes
        ]
        total_degree_sum = sum(node_degrees)

        return np.array([degree / total_degree_sum for degree in node_degrees])

    def g(self, G: nx.Graph, c: int) -> np.ndarray:
        """Auxiliar function to compute new particles"""
        n0_sum = np.sum(self.n0(G).flatten())
        n_sum = np.sum(self.n[c].flatten())
        p = self.probability_of_new_particles(G, c)
        return p * max(0, n0_sum - n_sum)

    def init(self, G: nx.Graph):
        """Initialize the LCU algorithm"""
        self.G = G
        self.n = self.n0(G)
        self.N = self.N0(G)
        self.delta = self.delta0(G)

    def step(self, G: nx.Graph):
        """Execute iteration of LCU algorithm"""
        for c in range(self.n_classes):
            self.P = self.probability(G)
            g_c = self.g(G, c)
            nc_product_pc = self.n[c] @ self.P[c]
            self.N[c] = np.diag(nc_product_pc)
            self.n[c] = nc_product_pc + g_c
            self.delta[c] += self.N[c]

    def unfold(self, G: nx.Graph) -> List[nx.Graph]:
        return [
            self.subnetwork_of_class(G, c)
            for c in self.classes
        ]

    def subnetwork_of_class(self, G: nx.Graph, c: int) -> nx.Graph:
        sub_graph = nx.Graph()
        sub_graph.add_nodes_from(G.nodes)
        domination = self.delta.copy()
        for c in range(self.n_classes):
            domination[c] = domination[c] + domination[c].T
        # FIXME: wrong application of argmax
        edges_subnetwork = [
            (i, j)
            for (i, j) in G.edges
            if np.argmax(domination[:, i, j], axis=0) + 1 == c
        ]
        sub_graph.add_edges_from(edges_subnetwork)

        return sub_graph


# alias
LCU = LabeledComponentUnfolding
