from typing import Dict, Callable, List, Optional
import numpy
import networkx
from loguru import logger
from egsis import complex_networks, features, lcu, superpixels, labeling
from egsis.graph_builder import GraphBuilder, GraphBuilderPlain

similarity_functions: Dict[str, Callable] = {
    "euclidian": features.euclidian_similarity,
    "euclidian_exp": features.euclidian_similarity_exp,
    "manhattan_exp": features.manhattan_similarity_exp,
    "manhattan_log": features.manhattan_similarity_log,
    "cosine": features.cosine_similarity,
}

class EGSIS:
    def __init__(
        self,
        superpixel_segments: int,
        superpixel_sigma: float,
        superpixel_compactness: float,
        feature_crop_image: bool = True,
        feature_extraction: features.FeaturesMethods = "comatrix",
        feature_similarity: str = "euclidian",
        graph_builder: Optional[GraphBuilder] = None,
        lcu_competition_level: float = 1,
        lcu_max_iter: int = 100
    ):
        self.superpixel_segments = superpixel_segments
        self.superpixel_sigma = superpixel_sigma
        self.superpixel_compactness = superpixel_compactness
        self.feature_extraction = feature_extraction
        self.feature_crop_image = feature_crop_image
        self.feature_similarity = similarity_functions[feature_similarity]
        self.graph_builder = graph_builder or GraphBuilderPlain()
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
        segments = segments - 1
        return segments

    def fit_predict(self, X: numpy.ndarray, y: numpy.ndarray):
        self.segments = self.build_superpixels(X)
        
        # Cria grafo temporário apenas para extrair features
        G_temp = complex_networks.complex_network_from_segments(self.segments)
        complex_networks.compute_node_features(G_temp, X, self.segments, self.feature_extraction)
        features = numpy.array([G_temp.nodes[n]["features"] for n in G_temp.nodes])
        
        # Constrói grafo final usando o builder (Plain ou KNN)
        self.G = self.graph_builder.build(self.segments, features)
        
        # Reinserir features calculadas na estrutura final
        for i, node in enumerate(self.G.nodes):
            self.G.nodes[node]["features"] = G_temp.nodes[node]["features"]
            
        complex_networks.compute_node_labels(self.G, self.segments, y)
        complex_networks.compute_edge_weights(self.G, self.feature_similarity)
        
        n_classes = len(numpy.unique(y)) - 1
        collective_dynamic = lcu.LabeledComponentUnfolding(
            competition_level=self.lcu_competition_level,
            max_iter=self.lcu_max_iter,
            n_classes=n_classes
        )
        self.sub_networks = collective_dynamic.fit_predict(self.G)
        return collective_dynamic.classify_vertexes(self.sub_networks)

    def fit_predict_segmentation_mask(self, X: numpy.ndarray, y: numpy.ndarray):
        G = self.fit_predict(X, y)
        superpixels_by_label = {node: G.nodes[node]["label"] for node in G.nodes}
        return labeling.create_segmentation_mask(self.segments, superpixels_by_label)
