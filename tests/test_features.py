import pytest
from egsis import features
import numpy

from skimage.data import cat as _cat
from skimage.util import img_as_ubyte


@pytest.fixture(scope="session")
def cat():
    return img_as_ubyte(_cat())


def test_feature_extraction(image_colorized):
    x = features.feature_extraction_comatrix(image_colorized)
    expected = numpy.array(
        [
            1., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            1., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            1., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
        ]
    )
    assert numpy.allclose(x, expected)


def test_crop_image(image_a):
    img = features.crop_image(image_a, max_radius=1, centroid=[1, 1])
    expected = numpy.array([[10, 20], [10, 20]])

    assert numpy.array_equal(img, expected)


def test_euclidian_distance(image_a, image_b):
    expected = 48.98979485566356
    assert features.euclidian_distance(image_a, image_b) == expected


def test_euclidian_similarity(image_a, image_b):
    expected = 0.020004082891064427
    assert features.euclidian_similarity(image_a, image_b) == expected


def test_cosine_similarity_success(image_a, image_b):
    expected = 0.7142857142857144
    assert features.cosine_similarity(image_a, image_b) == expected


def test_cosine_similarity_flatten(image_a, image_b):
    expected = 0.7142857142857144
    assert (
        features.cosine_similarity(
            image_a.flatten(),
            image_b.flatten(),
        )
        == expected
    )


def test_cosine_similarity_failure(image_a, image_b):

    with pytest.raises(ValueError):
        features.cosine_similarity(image_a, image_b.flatten())


def test_feature_extraction_segment(image_colorized, segments):
    x = features.feature_extraction_segment(
        img=image_colorized,
        segments=segments,
        label=2,
    )
    expected = numpy.array(
        [
            1., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            1., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            1., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
        ]
    )
    assert numpy.allclose(x, expected)


def test_get_segment_by_label(image_a, segments):
    x = features.get_segment_by_label(
        img=image_a,
        segments=segments,
        label=1,
    )
    expected = numpy.array(
        [
            [0, 0, 0],
            [10, 20, 30],
            [0, 0, 0],
        ]
    )
    assert numpy.array_equal(x, expected)


def test_get_segment_by_label_cropped(image_a, segments):
    x = features.get_segment_by_label_cropped(
        img=image_a,
        segments=segments,
        label=1,
        max_radius=1,
        centroid=[1, 1]
    )

    expected = numpy.array(
        [
            [0, 0],
            [10, 20]
        ]
    )
    assert numpy.array_equal(x, expected)


def test_feature_extraction_comatrix_should_have_one_dimension(image_a):
    feats = features._feature_extraction_comatrix_channel(image_a)
    assert len(feats.shape) == 1


def test_feature_extraction_comatrix_should_concatenate(image_a):
    img = numpy.array([image_a, image_a, image_a])
    feats = features.feature_extraction_comatrix(img)
    assert len(feats.shape) == 1


def test_crop_image_should_not_be_empty(cat):
    """Regression test: bug with empty crop image"""
    segment = features.crop_image(cat, max_radius=40, centroid=[18, 25])
    assert len(segment) != 0
