from typing import Dict, Callable, List

import numpy
import networkx
from loguru import logger

from egsis import complex_networks
from egsis import features
from egsis import lcu
from egsis import superpixels
from egsis import labeling


similarity_functions: Dict[str, Callable] = {
    "euclidian": features.euclidian_similarity,
    "euclidian_exp": features.euclidian_similarity_exp,
    "manhattan_exp": features.manhattan_similarity_exp,
    "manhattan_log": features.manhattan_similarity_log,
    "cosine": features.cosine_similarity,
}


class EGSIS:
    """
    [E]xploratory
    [G]raph-based
    [S]emi-supervised
    [I]mage
    [S]egmentation

    Notes
    -----

    Combines superpixels, complex networks and graph-based collective
    dynamics to solve a semi-supervised image segmentation challenge.

    Parameters
    ----------

    superpixel:
       - method: slic
       - segments
       - compactaness
       - gama

    complex networks:
       - network build method: superpixel neighbors

    feature extraction:
        - feature method: comatrix
        - crop image: True | False
        - erase_color
        - similarity function: euclidian | cosine

    labeled component unfolding:
        - competition_level
        - max_iter

    """

    def __init__(
        self,
        superpixel_segments: int,
        superpixel_sigma: float,
        superpixel_compactness: float,
        feature_crop_image: bool = True,
        feature_extraction: features.FeaturesMethods = "comatrix",
        feature_similarity: str = "euclidian",
        network_build_method: str = "neighbors",
        lcu_competition_level: float = 1,
        lcu_max_iter: int = 100
    ):
        self.superpixel_segments = superpixel_segments
        self.superpixel_sigma = superpixel_sigma
        self.superpixel_compactness = superpixel_compactness
        self.feature_extraction = feature_extraction
        self.feature_crop_image = feature_crop_image
        self.feature_similarity = similarity_functions[feature_similarity]
        self.network_build_method = network_build_method
        self.lcu_competition_level = lcu_competition_level
        self.lcu_max_iter = lcu_max_iter
        self.G: networkx.Graph
        self.sub_networks: List[networkx.Graph]
        self.segments: numpy.ndarray

    def build_superpixels(self, X) -> numpy.ndarray:
        segments = superpixels.build_superpixels_from_image(
            X,
            n_segments=self.superpixel_segments,
            compactness=self.superpixel_compactness,
            sigma=self.superpixel_sigma
        )
        # NOTE(@lerax): seg 01 mai 2023 09:47:57
        # segments indexing by 0, keep easier to control nodes vs edges vs numpy
        segments = segments - 1
        return segments

    def build_complex_network(
        self,
        X: numpy.ndarray,
        y: numpy.ndarray,
        segments: numpy.ndarray
    ) -> networkx.Graph:
        G = complex_networks.complex_network_from_segments(segments)
        complex_networks.compute_node_labels(
            graph=G,
            segments=segments,
            labels=y
        )
        logger.info("Complex networks: compute node labels finished.")
        complex_networks.compute_node_features(
            graph=G,
            img=X,
            segments=segments,
            feature_method=self.feature_extraction
        )
        logger.info("Complex networks: feature extraction finished.")
        complex_networks.compute_edge_weights(
            graph=G,
            similarity_function=self.feature_similarity
        )
        logger.info("Complex networks: compute node weights finished.")
        return G

    def sub_networks_to_matrix(self, sub_networks: List[networkx.Graph]):
        # FIXME: this should return subnetworks and a auxiliar
        # function should generate a new y_pred matrix
        return sub_networks

    def fit_predict(self, X: numpy.ndarray, y: numpy.ndarray):
        """

        Parameters:
        ------------
        X : numpy.ndarray (shape=(n, m, 3))
             it's the image matrix with values being the pixel
             luminosity of each color channel of RGB
        y : numpy.ndarray (shape=(n, m))
             it's the label matrix with partial annotation, to be full
             filled every non-zero value it's a label, and zero it's
             an unlabeled pixel.
        Returns
        -------
        new y matrix with full filled labels.
        """
        logger.info("Run!")
        self.segments = self.build_superpixels(X)
        logger.info("Superpixels: finished.")
        self.G = self.build_complex_network(X, y, self.segments)
        logger.info("Complex networks: finished.")
        n_classes = len(numpy.unique(y)) - 1
        collective_dynamic = lcu.LabeledComponentUnfolding(
            competition_level=self.lcu_competition_level,
            max_iter=self.lcu_max_iter,
            n_classes=n_classes
        )

        self.sub_networks = collective_dynamic.fit_predict(self.G)
        self.G_pred = collective_dynamic.classify_vertexes(self.sub_networks)
        logger.info("Dynamic collective LCU: finished.")

        # FIXME: should return a matrix y with new labels
        return self.G_pred

    def fit_predict_segmentation_mask(self, X: numpy.ndarray, y: numpy.ndarray):
        G = self.fit_predict(X, y)
        superpixels_by_label = {}
        for node in G.nodes:
            label = G.nodes[node]["label"]
            superpixels_by_label[node] = label

        return labeling.create_segmentation_mask(self.segments, superpixels_by_label)
