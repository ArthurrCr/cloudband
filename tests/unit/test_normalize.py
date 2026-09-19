import numpy as np

from cloudband.datasets.cloudsen12 import NODATA_VALUE
from cloudband.train.normalize import dynamic_z_score


def test_normalizes_each_channel_to_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    image = rng.normal(loc=5000, scale=800, size=(3, 64, 64)).astype(np.float32)

    normalized = dynamic_z_score(image)

    for channel in range(3):
        assert abs(normalized[channel].mean()) < 1e-3
        assert abs(normalized[channel].std() - 1.0) < 1e-3


def test_excludes_nodata_from_the_statistics():
    image = np.full((1, 4, 4), 100.0, dtype=np.float32)
    image[0, 0, 0] = NODATA_VALUE

    normalized = dynamic_z_score(image)

    valid_values = image[0][image[0] != NODATA_VALUE]
    assert valid_values.std() == 0
    assert np.all(normalized[0][image[0] != NODATA_VALUE] == 0.0)


def test_zeroes_nodata_pixels_in_the_output_instead_of_normalizing_them():
    rng = np.random.default_rng(1)
    image = rng.normal(loc=3000, scale=400, size=(2, 32, 32)).astype(np.float32)
    image[0, 5, 5] = NODATA_VALUE
    image[1, 10, 10] = NODATA_VALUE

    normalized = dynamic_z_score(image)

    assert normalized[0, 5, 5] == 0.0
    assert normalized[1, 10, 10] == 0.0


def test_a_fully_nodata_channel_does_not_raise():
    image = np.full((1, 4, 4), NODATA_VALUE, dtype=np.float32)
    normalized = dynamic_z_score(image)
    assert np.all(normalized == 0.0)