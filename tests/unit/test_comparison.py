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


def test_average_across_seeds_averages_matching_scenes():
    from cloudband.train.comparison import average_across_seeds

    seed_0 = pd.DataFrame({"clear": [80.0, 90.0]}, index=["s1", "s2"])
    seed_1 = pd.DataFrame({"clear": [90.0, 94.0]}, index=["s1", "s2"])

    averaged = average_across_seeds((seed_0, seed_1))

    assert averaged.loc["s1", "clear"] == pytest.approx(85.0)
    assert averaged.loc["s2", "clear"] == pytest.approx(92.0)


def test_average_across_seeds_stays_nan_when_every_seed_is_nan():
    from cloudband.train.comparison import average_across_seeds

    seed_0 = pd.DataFrame({"clear": [np.nan, 90.0]}, index=["s1", "s2"])
    seed_1 = pd.DataFrame({"clear": [np.nan, 94.0]}, index=["s1", "s2"])

    averaged = average_across_seeds((seed_0, seed_1))

    assert np.isnan(averaged.loc["s1", "clear"])
    assert averaged.loc["s2", "clear"] == pytest.approx(92.0)


def test_average_across_seeds_rejects_no_frames():
    from cloudband.train.comparison import average_across_seeds

    with pytest.raises(ValueError):
        average_across_seeds(())


def test_average_across_seeds_rejects_mismatched_experiments():
    from cloudband.train.comparison import average_across_seeds

    seed_0 = pd.DataFrame({"clear": [80.0], "cloud": [70.0]}, index=["s1"])
    seed_1 = pd.DataFrame({"clear": [80.0], "shadow": [70.0]}, index=["s1"])

    with pytest.raises(ValueError):
        average_across_seeds((seed_0, seed_1))


def test_seed_averaging_reaches_significance_that_naive_seed_pairing_cannot():
    """The money test: same underlying data, two ways of running Wilcoxon.

    Architecture A scores consistently higher than B across 20 scenes and
    5 seeds. Pairing directly by seed (N=5) cannot reach p < 0.05 no matter
    how consistent the difference is - the floor is 0.125. Averaging across
    seeds per scene first, then pairing by scene (N=20), can.
    """
    from cloudband.train.comparison import average_across_seeds, compare_per_scene_boa

    n_scenes = 20
    n_seeds = 5
    scene_ids = [f"s{i}" for i in range(n_scenes)]

    per_seed_frames_a = []
    per_seed_frames_b = []
    for seed in range(n_seeds):
        seed_rng = np.random.default_rng(seed)
        values_a = 85.0 + seed_rng.normal(0, 1.0, size=n_scenes)
        values_b = 80.0 + seed_rng.normal(0, 1.0, size=n_scenes)
        per_seed_frames_a.append(pd.DataFrame({"clear": values_a}, index=scene_ids))
        per_seed_frames_b.append(pd.DataFrame({"clear": values_b}, index=scene_ids))

    # the wrong way: average over scenes per seed, pair by seed (N=5)
    seed_level_a = pd.DataFrame(
        {"clear": [frame["clear"].mean() for frame in per_seed_frames_a]},
        index=[f"seed{i}" for i in range(n_seeds)],
    )
    seed_level_b = pd.DataFrame(
        {"clear": [frame["clear"].mean() for frame in per_seed_frames_b]},
        index=[f"seed{i}" for i in range(n_seeds)],
    )
    naive_result = compare_per_scene_boa(seed_level_a, seed_level_b)[0]

    # the right way: average over seeds per scene, pair by scene (N=20)
    averaged_a = average_across_seeds(tuple(per_seed_frames_a))
    averaged_b = average_across_seeds(tuple(per_seed_frames_b))
    correct_result = compare_per_scene_boa(averaged_a, averaged_b)[0]

    assert naive_result.n_pairs == n_seeds
    assert naive_result.p_value >= 0.0625

    assert correct_result.n_pairs == n_scenes
    assert correct_result.p_value < 0.05
    assert correct_result.p_value < naive_result.p_value


def test_compare_across_seeds_matches_the_two_step_version():
    from cloudband.train.comparison import average_across_seeds, compare_across_seeds

    seed_0_a = pd.DataFrame({"clear": [80.0, 90.0]}, index=["s1", "s2"])
    seed_1_a = pd.DataFrame({"clear": [82.0, 88.0]}, index=["s1", "s2"])
    seed_0_b = pd.DataFrame({"clear": [70.0, 75.0]}, index=["s1", "s2"])
    seed_1_b = pd.DataFrame({"clear": [72.0, 77.0]}, index=["s1", "s2"])

    combined = compare_across_seeds((seed_0_a, seed_1_a), (seed_0_b, seed_1_b))
    two_step = compare_per_scene_boa(
        average_across_seeds((seed_0_a, seed_1_a)),
        average_across_seeds((seed_0_b, seed_1_b)),
    )

    assert combined == two_step