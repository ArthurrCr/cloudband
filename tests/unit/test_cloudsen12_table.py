import math

import pytest

from cloudband.eval.confusion import BinaryConfusion
from cloudband.eval.metrics import balanced_overall_accuracy, producer_accuracy
from cloudband.eval.report import (
    CLOUDSEN12_EXPERIMENTS,
    cloudsen12_table,
    swap_positive,
)


def scene(clear, cloud, shadow):
    """Each argument is (tp, tn, fp, fn) for that experiment."""
    return {
        "clear": BinaryConfusion("clear", *clear),
        "cloud": BinaryConfusion("cloud", *cloud),
        "shadow": BinaryConfusion("shadow", *shadow),
    }


# a scene with cloud, shadow and clear pixels, all defined
WITH_CLOUD = scene(clear=(50, 40, 5, 5), cloud=(30, 60, 5, 5), shadow=(8, 85, 3, 4))
# a cloudless scene: no cloud pixels in the reference, but the model predicts some
NO_CLOUD = scene(clear=(90, 0, 0, 10), cloud=(0, 90, 10, 0), shadow=(0, 92, 8, 0))


def test_swap_positive_exchanges_the_two_classes():
    swapped = swap_positive(BinaryConfusion("clear", tp=1, tn=2, fp=3, fn=4), "invalid")

    assert (swapped.tp, swapped.tn, swapped.fp, swapped.fn) == (2, 1, 4, 3)
    assert swapped.label == "invalid"


def test_table_has_the_three_paper_experiments_in_order():
    table = cloudsen12_table({"s1": WITH_CLOUD})

    assert list(table.index) == list(CLOUDSEN12_EXPERIMENTS)
    assert table.index.name == "experiment"
    assert {"median_boa", "pa_low", "ua_high", "n_patches", "n_pa", "n_ua"} <= set(
        table.columns
    )


def test_ua_is_dropped_in_every_experiment_for_a_scene_without_cloud():
    table = cloudsen12_table({"s1": WITH_CLOUD, "s2": NO_CLOUD})

    # s2 has no cloud: its UA must not count, in any experiment, even though the
    # shadow and valid/invalid UA would otherwise be defined there
    assert table.loc["cloud/no cloud", "n_ua"] == 1
    assert table.loc["cloud shadow", "n_ua"] == 1
    assert table.loc["valid/invalid", "n_ua"] == 1


def test_pa_is_dropped_only_where_the_experiment_class_is_absent():
    table = cloudsen12_table({"s1": WITH_CLOUD, "s2": NO_CLOUD})

    assert table.loc["cloud/no cloud", "n_pa"] == 1   # s2 has no cloud
    assert table.loc["cloud shadow", "n_pa"] == 1     # s2 has no shadow
    assert table.loc["valid/invalid", "n_pa"] == 1    # s2 has no invalid pixel


def test_valid_invalid_uses_invalid_as_the_positive_class():
    table = cloudsen12_table({"s1": WITH_CLOUD})
    clear = WITH_CLOUD["clear"]

    # PA of invalid is the specificity of clear: tn / (tn + fp) = 40 / 45
    invalid_pa = producer_accuracy(swap_positive(clear, "invalid"))
    assert invalid_pa == pytest.approx(40 / 45)
    # 0.889 falls in the middle range, whereas PA of clear (50/55 = 0.909) is high
    assert table.loc["valid/invalid", "pa_middle"] == 100.0
    assert table.loc["valid/invalid", "pa_high"] == 0.0


def test_valid_invalid_median_boa_equals_that_of_clear():
    # BOA is symmetric in the two classes, so swapping them cannot change it
    other = scene((70, 20, 5, 5), (30, 60, 5, 5), (8, 85, 3, 4))
    per_scene = {"s1": WITH_CLOUD, "s2": other}
    table = cloudsen12_table(per_scene)

    boas = sorted(balanced_overall_accuracy(s["clear"]) for s in per_scene.values())
    expected = round(100 * (boas[0] + boas[1]) / 2, 2)
    assert table.loc["valid/invalid", "median_boa"] == pytest.approx(expected)


def test_n_patches_counts_scenes_with_a_defined_boa():
    all_cloud = scene(clear=(0, 0, 0, 0), cloud=(100, 0, 0, 0), shadow=(0, 100, 0, 0))
    table = cloudsen12_table({"s1": WITH_CLOUD, "s2": NO_CLOUD, "s3": all_cloud})

    # cloud BOA needs cloud AND non-cloud: s2 has no cloud, s3 has no non-cloud
    assert table.loc["cloud/no cloud", "n_patches"] == 1
    assert math.isfinite(table.loc["cloud/no cloud", "median_boa"])