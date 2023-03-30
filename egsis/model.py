from typing import Dict, Callable, List, Literal

import numpy
import networkx
from loguru import logger

from egsis import complex_networks
from egsis import features
from egsis import lcu
from egsis import superpixels


similarity_functions: Dict[str, Callable] = {
    "euclidian": features.euclidian_similarity,
    "cosine": features.cosine_similarity
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
        - feature method: multidimensional fast fourier transform
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
        feature_extraction: Literal["fft", "comatrix"] = "fft",
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

    def build_superpixels(self, X) -> numpy.ndarray:
        segments = superpixels.build_superpixels_from_image(
            X,
            n_segments=self.superpixel_segments,
            compactness=self.superpixel_compactness,
            sigma=self.superpixel_sigma
        )
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
        complex_networks.compute_node_features(
            graph=G,
            img=X,
            segments=segments,
            feature_method=self.feature_extraction
        )
        complex_networks.compute_edge_weights(
            graph=G,
            similarity_function=self.feature_similarity
        )
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
             filled Every non-zero value it's a label, and zero it's
             an unlabeled pixel.
        Returns
        -------
        new y matrix with full filled labels.
        """
        logger.info("Run!")
        segments = self.build_superpixels(X)
        G = self.build_complex_network(X, y, segments)
        n_classes = numpy.unique(y).count() - 1
        collective_dynamic = lcu.LabeledComponentUnfolding(
            competition_level=self.lcu_competition_level,
            max_iter=self.lcu_max_iter,
            n_classes=n_classes
        )

        sub_networks = collective_dynamic.fit_predict(G)

        # FIXME: should return a matrix y with new labels
        return collective_dynamic.classify_vertexes(sub_networks)
