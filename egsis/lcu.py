from typing import List

import networkx as nx

from egsis import features

similarity_functions = {
    "euclidian": features.euclidian_distance,
    "cosine": features.cosine_similarity
}


class LabelComponentUnfolding:

    """
    Collective Dynamic Label Component Unfolding


    It can be used to solve Semi-Supervised problems.


    Parameters
    ----------
    n_classes : int
         Number of classes of a given problem. Use n_classes=2 for binary
         classification, for instance.

    competition_level : float
         (lambda) varies between 0 and 1.
          0 means random walk and 1 max competition (recommended).

    max_iter : int
         The time limit parameter controls when the simulation should
         stop; it must be at least as large as the diameter of the network.

    similarity: str
         Similarity function name to be used in vertex walk
         probability function.

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
        similarity='euclidian'
    ):
        assert competition_level > 0 and competition_level <= 1
        self.competition_level = competition_level
        self.max_iter = max_iter
        self.n_classes = n_classes
        self.similarity_function = similarity_functions[similarity]
        self.complex_network = None

    def fit_predict(self, complex_network: nx.Graph) -> nx.Graph:
        """Fit complex network and predict new unlabeled data points.

        Each vertex of the complex_network should contains the
        label (if it's labeled) and it's features.
        """
        pass

    def unfold(self, complex_network: nx.Graph) -> List[nx.Graph]:
        pass

    def current_relative_subordination(self, node, label):
        pass

    def current_directed_domination(self, node, label):
        pass

    def cumulative_domination(self, node, label):
        pass

    def subnetwork_of_class(self, label):
        pass

    def probability_function(self):
        pass
