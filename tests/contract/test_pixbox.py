"""Contract tests for the PixBox Sentinel-2 reference labels.

Counts are pinned to the reference collection. If a distributed file disagrees,
verify_label_counts reports the difference instead of the pipeline silently
producing plausible wrong numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cloudband.eval.experiments import prediction_masks, run_experiments
from cloudband.labels.pixbox import (
    CLOUD_COLUMN,
    CLOUD_IDS,
    COLUMN_COLUMN,
    EXPECTED_CLOUD_SHADOW_OVERLAP,
    EXPECTED_LABEL_COUNTS,
    NOT_CLOUD_IDS,
    NOT_SHADOW_IDS,
    PRODUCT_COLUMN,
    PRODUCT_ID_TO_SCENE,
    ROW_COLUMN,
    SHADOW_COLUMN,
    SHADOW_IDS,
    TOTAL_PIXELS,
    cloud_shadow_overlap,
    label_counts,
    label_masks,
    sample_predictions,
    verify_label_counts,
)

SCENE_COUNT = 29


def build_table(rows: list[tuple[int, int]]) -> pd.DataFrame:
    """Build a minimal table from (cloud id, shadow id) pairs."""
    return pd.DataFrame(
        {
            CLOUD_COLUMN: [cloud for cloud, _ in rows],
            SHADOW_COLUMN: [shadow for _, shadow in rows],
            PRODUCT_COLUMN: [next(iter(PRODUCT_ID_TO_SCENE))] * len(rows),
            ROW_COLUMN: list(range(len(rows))),
            COLUMN_COLUMN: [0] * len(rows),
        }
    )


def test_scene_manifest_is_complete_and_unique() -> None:
    assert len(PRODUCT_ID_TO_SCENE) == SCENE_COUNT
    assert len(set(PRODUCT_ID_TO_SCENE.values())) == SCENE_COUNT
    assert all(
        name.startswith(("S2A_MSIL1C", "S2B_MSIL1C")) for name in PRODUCT_ID_TO_SCENE.values()
    )


def test_cloud_id_sets_are_disjoint_and_cover_the_range() -> None:
    assert not CLOUD_IDS & NOT_CLOUD_IDS
    assert sorted(CLOUD_IDS | NOT_CLOUD_IDS) == list(range(13))


def test_shadow_id_sets_are_disjoint() -> None:
    assert not SHADOW_IDS & NOT_SHADOW_IDS


def test_expected_counts_are_arithmetically_consistent() -> None:
    """Clear is the conjunction of both negatives, so the three do not partition."""
    not_cloud = TOTAL_PIXELS - EXPECTED_LABEL_COUNTS["cloud"]
    implied_overlap = EXPECTED_LABEL_COUNTS["shadow"] - (not_cloud - EXPECTED_LABEL_COUNTS["clear"])
    assert implied_overlap == EXPECTED_CLOUD_SHADOW_OVERLAP
    naive_clear = TOTAL_PIXELS - EXPECTED_LABEL_COUNTS["cloud"] - EXPECTED_LABEL_COUNTS["shadow"]
    assert EXPECTED_LABEL_COUNTS["clear"] > naive_clear


def test_cloud_and_shadow_can_both_be_true() -> None:
    table = build_table([(6, 3)])
    masks = label_masks(table)
    assert masks["cloud"][0]
    assert masks["shadow"][0]
    assert not masks["clear"][0]


def test_clear_requires_both_negatives() -> None:
    table = build_table([(0, 0), (0, 3), (6, 0)])
    masks = label_masks(table)
    assert masks["clear"].tolist() == [True, False, False]


def test_topographic_shadow_is_not_cloud_shadow() -> None:
    table = build_table([(0, 4), (0, 2)])
    masks = label_masks(table)
    assert not masks["shadow"].any()
    assert masks["clear"].all()


def test_label_counts_sum_can_exceed_row_count() -> None:
    table = build_table([(6, 3), (0, 0)])
    counts = label_counts(label_masks(table))
    assert sum(counts.values()) == 3
    assert len(table) == 2


def test_overlap_is_counted() -> None:
    table = build_table([(6, 3), (6, 3), (0, 0)])
    assert cloud_shadow_overlap(label_masks(table)) == 2


def test_verify_reports_differences_against_reference() -> None:
    table = build_table([(0, 0)])
    differences = verify_label_counts(table)
    assert differences["total"] == 1 - TOTAL_PIXELS
    assert differences["cloud"] == -EXPECTED_LABEL_COUNTS["cloud"]


def test_missing_column_is_rejected() -> None:
    table = pd.DataFrame({CLOUD_COLUMN: [0]})
    with pytest.raises(ValueError, match="missing columns"):
        label_masks(table)


def test_predictions_are_sampled_at_pixel_indices() -> None:
    table = build_table([(0, 0), (0, 0)])
    table[ROW_COLUMN] = [2, 0]
    table[COLUMN_COLUMN] = [1, 3]
    prediction = np.arange(16).reshape(4, 4)
    assert sample_predictions(table, prediction).tolist() == [9, 3]


def test_out_of_bounds_indices_are_rejected() -> None:
    table = build_table([(0, 0)])
    table[ROW_COLUMN] = [99]
    with pytest.raises(ValueError, match="outside the"):
        sample_predictions(table, np.zeros((4, 4), dtype=int))


def test_thin_cloud_folds_into_cloud() -> None:
    masks = prediction_masks(np.array([0, 1, 2, 3]))
    assert masks["cloud"].tolist() == [False, True, True, False]
    assert masks["clear"].tolist() == [True, False, False, False]
    assert masks["shadow"].tolist() == [False, False, False, True]


def test_unknown_prediction_code_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown predicted class codes"):
        prediction_masks(np.array([0, 4]))


def test_experiments_run_on_overlapping_reference_labels() -> None:
    table = build_table([(6, 3), (0, 0), (6, 0)])
    reference = label_masks(table)
    predicted = prediction_masks(np.array([1, 0, 3]))
    confusions = run_experiments(reference, predicted)
    assert set(confusions) == {"clear", "cloud", "shadow"}
    assert confusions["cloud"].tp == 1
    assert confusions["cloud"].fn == 1
    assert confusions["shadow"].fn == 1