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
    ):
        assert competition_level > 0 and competition_level <= 1
        self.competition_level = competition_level
        self.max_iter = max_iter
        self.n_classes = n_classes
        self.classes = list(range(1, n_classes + 1))
        self.n: np.ndarray
        self.N: np.ndarray
        self.delta: np.ndarray

    def n0(self, G: nx.Graph, c: int):
        pass

    def N0(self, G: nx.Graph, c: int):
        pass

    def delta0(self, G: nx.Graph, c: int, competition_level: float):
        pass

    def g(self, G: nx.Graph, N, competition_level: float):
        pass

    def init(self, G: nx.Graph):
        C = self.n_classes
        i = j = G.number_of_nodes
        self.n = np.zeros(shape=(C, i))
        self.N = np.zeros(shape=(C, i, j))
        self.delta = np.zeros(shape=(C, i, j))
        for c in self.classes:
            self.n[c] = self.n0(G, c)
            self.N[c] = self.N0(G, c)
            self.delta[c] = self.delta0(G, c, self.competition_level)

    def step(self, G: nx.Graph, t: int):
        for c in self.classes:
            P = self.probability_function(G, self.N)
            g_c = self.g(G, self.N, self.competition_level)
            self.N[c] = self.N0(G, c)
            self.n[c] = self.n[c] @ P + g_c
            self.delta[c] += self.N[c]

    def fit_predict(self, G: nx.Graph) -> nx.Graph:
        """Fit complex network and predict new unlabeled data points.

        Each vertex of the complex_network should contains the
        label (if it's labeled) and it's features.
        """
        self.init(G)
        for t in range(self.max_iter):
            self.step(G, t)

    def unfold(self, G: nx.Graph) -> List[nx.Graph]:
        pass

    def current_relative_subordination(
        self,
        G: nx.Graph,
        node: int,
        label: int
    ):
        pass

    def current_directed_domination(self, G: nx.Graph, node: int, label: int):
        pass

    def cumulative_domination(self, G: nx.Graph, node: int, label: int):
        pass

    def subnetwork_of_class(self, G: nx.Graph, label: int) -> nx.Graph:
        pass

    def probability_function(self, G: nx.Graph, N: np.ndarray) -> np.ndarray:
        pass
