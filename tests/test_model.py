import pytest

from skimage.data import cat
from skimage.util import img_as_ubyte

from egsis import model
from egsis.superpixels import build_superpixels_from_image
from egsis.labeling import create_label_matrix


@pytest.fixture
def egsis_model():
    return model.EGSIS(
        superpixel_segments=53,
        superpixel_sigma=0.3,
        superpixel_compactness=40,
        feature_extraction="comatrix",
        lcu_max_iter=25,
    )


@pytest.fixture
def X():
    return img_as_ubyte(cat())


@pytest.fixture
def y(X):
    superpixels = build_superpixels_from_image(
        X,
        n_segments=53,
        compactness=40,
        sigma=0.3
    )
    superpixel_labels = {
        1: 1,
        4: 1,
        43: 1,
        44: 1,
        41: 2,
        21: 2,
        35: 2,
        22: 2,
        40: 2,
        23: 2,
    }
    y = create_label_matrix(shape=superpixels.shape)
    for superpixel, label in superpixel_labels.items():
        y[superpixels == superpixel] = label
    return y


@pytest.fixture
def segments(X, egsis_model):
    return egsis_model.build_superpixels(X)


def test_egsis_build_complex_network(egsis_model, X, y, segments):
    G = egsis_model.build_complex_network(X, y, segments)
    assert all(G.nodes[n].get("features").any() for n in G.nodes)
    assert all(G.edges[e].get("weight") for e in G.edges)
    assert set(G.nodes[n].get("label") for n in G.nodes) == {0, 1, 2}


def test_egsis_fit_predict(egsis_model, X, y):
    G = egsis_model.fit_predict(X, y)

    assert set(G.nodes[n]["label"] for n in G.nodes) == {1, 2}
