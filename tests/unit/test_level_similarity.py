import math

import numpy as np
import pandas as pd
import pytest

from cloudband.datasets.cloudsen12 import NODATA_VALUE, Sample
from cloudband.eval.level_similarity import (
    compare_channel,
    compare_levels,
    level_pairs,
    summarize,
)


def surface(seed, shape=(32, 32)):
    rng = np.random.default_rng(seed)
    return rng.uniform(500, 4000, size=shape)


def test_an_affine_change_disappears_after_normalization():
    l1c = surface(0)
    l2a = 0.8 * l1c - 300

    result = compare_channel(l1c, l2a)

    assert result["raw_relative_difference"] > 0.2
    assert result["correlation"] == pytest.approx(1.0)
    assert result["z_rmse"] == pytest.approx(0.0, abs=1e-5)


def test_unrelated_images_stay_apart_after_normalization():
    result = compare_channel(surface(0, (200, 200)), surface(1, (200, 200)))

    assert abs(result["correlation"]) < 0.05
    assert result["z_rmse"] == pytest.approx(math.sqrt(2), abs=0.05)


def test_normalized_difference_follows_the_correlation():
    l1c = surface(0)
    l2a = l1c + surface(1) * 0.5

    result = compare_channel(l1c, l2a)

    assert result["z_rmse"] == pytest.approx(math.sqrt(2 * (1 - result["correlation"])))


def test_nodata_pixels_are_left_out_of_both_images():
    l1c = surface(0)
    l2a = 0.8 * l1c - 300
    l2a[:4, :4] = NODATA_VALUE
    l1c[-4:, -4:] = NODATA_VALUE

    result = compare_channel(l1c, l2a)

    assert result["correlation"] == pytest.approx(1.0)
    assert result["z_rmse"] == pytest.approx(0.0, abs=1e-5)


def test_a_flat_channel_has_no_correlation():
    result = compare_channel(np.full((8, 8), 1000.0), surface(0, (8, 8)))

    assert math.isnan(result["correlation"])


def test_level_pairs_match_by_id_and_shift_the_l2a_positions():
    l1c = pd.DataFrame({"tortilla:id": ["a", "b", "c"]})
    l2a = pd.DataFrame({"tortilla:id": ["c", "a"]})

    assert level_pairs(l1c, l2a, l2a_offset=3) == [(0, 4), (2, 3)]


def test_compare_levels_gives_one_row_per_patch_and_band():
    images = [surface(i, (3, 16, 16)) for i in range(2)]
    stack = images + [0.9 * image + 50 for image in images]
    ids = ["a", "b", "a", "b"]
    table = pd.DataFrame({"tortilla:id": ids})

    def read(table, index, crop_to_valid=True):
        return Sample(identifier=ids[index], image=stack[index], annotation=np.zeros((16, 16)))

    comparison = compare_levels([(0, 2), (1, 3)], table, read)

    assert len(comparison) == 6
    assert set(comparison.band) == {"B04", "B03", "B8A"}
    assert np.allclose(comparison.z_rmse, 0.0, atol=1e-5)
    assert list(summarize(comparison).index) == ["B04", "B03", "B8A"]


def test_no_pairs_gives_an_empty_table_that_still_summarizes():
    comparison = compare_levels([], pd.DataFrame({"tortilla:id": []}), lambda *args: None)

    assert comparison.empty
    assert summarize(comparison).empty
