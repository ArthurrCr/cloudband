import numpy as np
import pandas as pd
import pytest

from cloudband.train.comparison import compare_per_scene_boa


def test_rejects_frames_scoring_different_experiments():
    boa_a = pd.DataFrame({"clear": [1.0], "cloud": [1.0]})
    boa_b = pd.DataFrame({"clear": [1.0], "shadow": [1.0]})
    with pytest.raises(ValueError):
        compare_per_scene_boa(boa_a, boa_b)


def test_drops_scenes_undefined_in_either_frame():
    boa_a = pd.DataFrame(
        {"clear": [90.0, 92.0, np.nan]}, index=["s1", "s2", "s3"]
    )
    boa_b = pd.DataFrame(
        {"clear": [91.0, np.nan, 95.0]}, index=["s1", "s2", "s3"]
    )

    results = compare_per_scene_boa(boa_a, boa_b)

    assert len(results) == 1
    assert results[0].experiment == "clear"
    assert results[0].n_pairs == 1


def test_pairs_by_scene_identifier_not_row_position():
    boa_a = pd.DataFrame({"clear": [10.0, 20.0]}, index=["s2", "s1"])
    boa_b = pd.DataFrame({"clear": [1.0, 2.0]}, index=["s1", "s2"])
    # by identifier: s1 -> a=20.0, b=1.0, diff=19.0; s2 -> a=10.0, b=2.0, diff=8.0
    # a position-based pairing would instead give diffs of 9.0 and 18.0

    results = compare_per_scene_boa(boa_a, boa_b)

    assert results[0].n_pairs == 2
    assert results[0].median_difference == pytest.approx(13.5)


def test_identical_frames_give_a_p_value_of_one():
    boa_a = pd.DataFrame({"clear": [80.0, 85.0, 90.0]}, index=["s1", "s2", "s3"])
    boa_b = boa_a.copy()

    results = compare_per_scene_boa(boa_a, boa_b)

    assert results[0].p_value == 1.0
    assert results[0].median_difference == 0.0


def test_no_pairable_scenes_gives_nan_result():
    boa_a = pd.DataFrame({"clear": [np.nan, np.nan]}, index=["s1", "s2"])
    boa_b = pd.DataFrame({"clear": [80.0, 85.0]}, index=["s1", "s2"])

    results = compare_per_scene_boa(boa_a, boa_b)

    assert results[0].n_pairs == 0
    assert np.isnan(results[0].p_value)


def test_reports_one_result_per_experiment_column():
    boa_a = pd.DataFrame(
        {"clear": [90.0], "cloud": [88.0], "shadow": [70.0]}, index=["s1"]
    )
    boa_b = pd.DataFrame(
        {"clear": [91.0], "cloud": [87.0], "shadow": [72.0]}, index=["s1"]
    )

    results = compare_per_scene_boa(boa_a, boa_b)

    assert {r.experiment for r in results} == {"clear", "cloud", "shadow"}