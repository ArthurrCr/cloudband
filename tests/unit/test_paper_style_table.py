import pandas as pd
import pytest

from cloudband.eval.report import (
    bucket_percentages,
    median_metric,
    paper_style_table,
)


def test_bucket_percentages_sorts_values_into_the_three_fixed_ranges():
    # 5 scenes: 0.05 -> low, 0.1 and 0.5 and 0.9 -> middle, 0.95 -> high
    per_scene = pd.DataFrame(
        {"clear": [0.05, 0.1, 0.5, 0.9, 0.95]},
    )

    result = bucket_percentages(per_scene)

    assert result.loc["clear", "low"] == pytest.approx(20.0)
    assert result.loc["clear", "middle"] == pytest.approx(60.0)
    assert result.loc["clear", "high"] == pytest.approx(20.0)


def test_bucket_percentages_sum_to_100_when_every_scene_has_a_value():
    per_scene = pd.DataFrame({"clear": [0.02, 0.3, 0.6, 0.88, 0.99, 0.0, 1.0]})

    result = bucket_percentages(per_scene)

    row = result.loc["clear"]
    total = row["low"] + row["middle"] + row["high"]
    assert total == pytest.approx(100.0)


def test_bucket_percentages_excludes_undefined_scenes_from_the_total():
    # a NaN scene (class absent from that scene) must not count toward the total
    per_scene = pd.DataFrame({"clear": [0.05, 0.05, float("nan"), float("nan")]})

    result = bucket_percentages(per_scene)

    # both real values are low; NaNs excluded, so low% is 100, not 50
    assert result.loc["clear", "low"] == pytest.approx(100.0)
    assert result.loc["clear", "middle"] == pytest.approx(0.0)
    assert result.loc["clear", "high"] == pytest.approx(0.0)


def test_bucket_percentages_gives_nan_when_no_scene_has_a_value():
    per_scene = pd.DataFrame({"clear": [float("nan"), float("nan")]})

    result = bucket_percentages(per_scene)

    assert result.loc["clear", "low"] != result.loc["clear", "low"]  # NaN


def test_median_metric_matches_pandas_median():
    per_scene = pd.DataFrame({"clear": [0.8, 0.9, 0.95], "cloud": [0.7, 0.75, 0.9]})

    result = median_metric(per_scene)

    assert result["clear"] == pytest.approx(0.9)
    assert result["cloud"] == pytest.approx(0.75)


def test_paper_style_table_has_the_papers_own_columns():
    per_scene_boa = pd.DataFrame(
        {"clear": [0.9, 0.94, 0.96], "cloud": [0.88, 0.9, 0.92]}
    )
    per_scene_pa = pd.DataFrame({"clear": [0.05, 0.5, 0.95], "cloud": [0.2, 0.6, 0.99]})
    per_scene_ua = pd.DataFrame({"clear": [0.05, 0.5, 0.95], "cloud": [0.2, 0.6, 0.99]})
    n_patches = {"clear": 3, "cloud": 3}

    table = paper_style_table(per_scene_boa, per_scene_pa, per_scene_ua, n_patches)

    expected_columns = {
        "median_boa",
        "pa_low",
        "pa_middle",
        "pa_high",
        "ua_low",
        "ua_middle",
        "ua_high",
        "n_patches",
    }
    assert set(table.columns) == expected_columns
    assert table.loc["clear", "median_boa"] == pytest.approx(94.0)
    assert table.loc["clear", "n_patches"] == 3


def test_paper_style_table_index_is_named_experiment_regardless_of_input_naming():
    # deliberately unnamed columns, the way a plain pd.DataFrame({...}) gives by
    # default, to catch the index name silently not propagating
    per_scene_boa = pd.DataFrame({"clear": [0.9, 0.9]})
    per_scene_pa = pd.DataFrame({"clear": [0.5, 0.5]})
    per_scene_ua = pd.DataFrame({"clear": [0.5, 0.5]})

    table = paper_style_table(per_scene_boa, per_scene_pa, per_scene_ua, {"clear": 2})

    assert table.index.name == "experiment"


def test_paper_style_table_round_trips_through_csv(tmp_path):
    per_scene_boa = pd.DataFrame({"clear": [0.9, 0.94, 0.96]})
    per_scene_pa = pd.DataFrame({"clear": [0.95, 0.5, 0.05]})
    per_scene_ua = pd.DataFrame({"clear": [0.95, 0.5, 0.05]})

    table = paper_style_table(per_scene_boa, per_scene_pa, per_scene_ua, {"clear": 3})
    path = tmp_path / "paper_table.csv"
    table.to_csv(path)

    reloaded = pd.read_csv(path, index_col="experiment")
    assert (table.round(6) == reloaded.round(6)).all().all()


def test_paper_style_table_median_boa_is_a_percentage_not_a_fraction():
    per_scene_boa = pd.DataFrame({"clear": [0.9, 0.9, 0.9]})
    per_scene_pa = pd.DataFrame({"clear": [0.5, 0.5, 0.5]})
    per_scene_ua = pd.DataFrame({"clear": [0.5, 0.5, 0.5]})

    table = paper_style_table(per_scene_boa, per_scene_pa, per_scene_ua, {"clear": 3})

    # 0.9 as a fraction must show as 90.0, not 0.9 - a common unit-mismatch bug
    assert table.loc["clear", "median_boa"] == pytest.approx(90.0)