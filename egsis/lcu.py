from typing import List

import networkx as nx


class LabelComponentUnfolding:

    """
    Collective Dynamic Label Component Unfolding


    It can be used to solve Semi-Supervised problems.


    Parameters
    ----------
    competition_level: float
         (lambda) varies between 0 and 1.
          0 means random walk and 1 max competition (recommended).
    tau: int
         The time limit parameter tau controls when the simulation should
         stop; it must be at least as large as the diameter of the network.

    Reference
    ---------
    Filipe Alves Neto Verri, Paulo Roberto Urio, Liang Zhao: “Network
    Unfolding Map by Edge Dynamics Modeling”, 2016, IEEE Transactions
    on Neural Networks and Learning Systems, vol. 29, no. 2,
    pp. 405-418, Feb. 2018. doi: 10.1109/TNNLS.2016.2626341;
    arXiv:1603.01182. DOI: 10.1109/TNNLS.2016.2626341.
    """

    def __init__(self, competition_level: float = 1, tau: int = 500):
        assert competition_level > 0 and competition_level <= 1
        self.competition_level = competition_level
        self.tau = tau

    def fit_predict(self, complex_network: nx.Graph) -> nx.Graph:
        """Fit complex network and predict new unlabeled data points.

        The each vertex of the complex_network should contains the
        label (if it's labeled) and it's features.
        """
        pass

    def unfold(self, G: nx.Graph) -> List[nx.Graph]:
        pass
