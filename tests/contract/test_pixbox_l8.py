"""Contract tests for the PixBox Landsat 8 labels and scoring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cloudband.labels import pixbox_l8 as labels
from cloudband.pipelines import pixbox_l8 as pipeline

SCENE_COUNT = 11
FIRST_PRODUCT = next(iter(labels.PRODUCT_ID_TO_SCENE))


def build_table(rows: list[tuple[int, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            labels.SURFACE_COLUMN: [surface for surface, _ in rows],
            labels.SHADOW_COLUMN: [shadow for _, shadow in rows],
            labels.PRODUCT_COLUMN: [FIRST_PRODUCT] * len(rows),
            labels.ROW_COLUMN: list(range(len(rows))),
            labels.COLUMN_COLUMN: [0] * len(rows),
        }
    )


def test_scene_manifest_is_complete_and_unique() -> None:
    assert len(labels.PRODUCT_ID_TO_SCENE) == SCENE_COUNT
    assert len(set(labels.PRODUCT_ID_TO_SCENE.values())) == SCENE_COUNT
    assert all(name.startswith("LC8") for name in labels.PRODUCT_ID_TO_SCENE.values())


def test_cloud_lives_in_surface_type_not_cloud_characteristics() -> None:
    """CLOUD_CHARACTERISTICS_ID here encodes cloud type, not presence."""
    assert labels.SURFACE_COLUMN == "PIXEL_SURFACE_TYPE_ID"
    assert labels.CLOUD_TYPE_COLUMN not in (labels.SURFACE_COLUMN, labels.SHADOW_COLUMN)


def test_surface_id_sets_are_disjoint() -> None:
    sets = (
        labels.SURFACE_CLOUD_IDS,
        labels.SURFACE_NOT_CLOUD_IDS,
        labels.SURFACE_MIXED_CLOUD_IDS,
        labels.SURFACE_OTHER_IDS,
    )
    union: set[int] = set()
    for group in sets:
        assert not union & group
        union |= group
    assert sorted(union) == list(range(14))


def test_shadow_is_a_plain_flag() -> None:
    """Unlike Sentinel-2, this collection does not separate shadow kinds."""
    assert sorted(labels.SHADOW_IDS) == [1]
    assert sorted(labels.NOT_SHADOW_IDS) == [0]


def test_expected_counts_match_the_published_matrices() -> None:
    """Reference positives implied by OmniCloudMask Table A.2."""
    assert labels.EXPECTED_LABEL_COUNTS["cloud"] == 4586 + 892
    assert labels.EXPECTED_LABEL_COUNTS["shadow"] == 710 + 686
    assert labels.EXPECTED_LABEL_COUNTS["clear"] == 12290 + 75


def test_expected_counts_are_arithmetically_consistent() -> None:
    not_cloud = labels.TOTAL_PIXELS - labels.EXPECTED_LABEL_COUNTS["cloud"]
    implied = labels.EXPECTED_LABEL_COUNTS["shadow"] - (
        not_cloud - labels.EXPECTED_LABEL_COUNTS["clear"]
    )
    assert implied == labels.EXPECTED_CLOUD_SHADOW_OVERLAP


def test_cloud_and_shadow_can_both_be_true() -> None:
    masks = labels.label_masks(build_table([(0, 1)]))
    assert masks["cloud"][0]
    assert masks["shadow"][0]
    assert not masks["clear"][0]


def test_clear_requires_both_negatives() -> None:
    masks = labels.label_masks(build_table([(3, 0), (3, 1), (0, 0)]))
    assert masks["clear"].tolist() == [True, False, False]


def test_semi_transparent_cloud_counts_as_cloud() -> None:
    masks = labels.label_masks(build_table([(1, 0)]))
    assert masks["cloud"][0]


def test_mixed_snow_ice_water_is_not_cloud() -> None:
    masks = labels.label_masks(build_table([(12, 0)]))
    assert not masks["cloud"][0]
    assert masks["clear"][0]


def test_unmapped_surface_value_is_rejected() -> None:
    with pytest.raises(ValueError, match="unmapped values"):
        labels.label_masks(build_table([(99, 0)]))


def test_missing_column_is_rejected() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        labels.label_masks(pd.DataFrame({labels.SURFACE_COLUMN: [0]}))


def test_verify_reports_differences() -> None:
    differences = labels.verify_label_counts(build_table([(3, 0)]))
    assert differences["total"] == 1 - labels.TOTAL_PIXELS
    assert differences["cloud"] == -labels.EXPECTED_LABEL_COUNTS["cloud"]


def test_landsat_nir_is_band_five() -> None:
    """Sentinel-2 NIR is B8 or B8A; the OLI NIR is B5."""
    assert pipeline.RGN_BANDS == (4, 3, 5)
    assert pipeline.NATIVE_RESOLUTION_M == 30


def test_predictions_are_sampled_at_pixel_indices() -> None:
    table = build_table([(3, 0), (3, 0)])
    table[labels.ROW_COLUMN] = [2, 0]
    table[labels.COLUMN_COLUMN] = [1, 3]
    prediction = np.arange(16).reshape(4, 4)
    assert labels.sample_predictions(table, prediction).tolist() == [9, 3]


def test_out_of_bounds_indices_are_rejected() -> None:
    table = build_table([(3, 0)])
    table[labels.ROW_COLUMN] = [99]
    with pytest.raises(ValueError, match="outside the"):
        labels.sample_predictions(table, np.zeros((4, 4), dtype=int))


def test_score_requires_attached_predictions() -> None:
    with pytest.raises(ValueError, match="call attach_predictions first"):
        pipeline.score(build_table([(3, 0)]))


def test_unknown_product_id_is_rejected(tmp_path: Path) -> None:
    table = build_table([(3, 0)])
    table[labels.PRODUCT_COLUMN] = [-1]
    with pytest.raises(ValueError, match="no scene name known"):
        pipeline.attach_predictions(table, tmp_path)


def test_missing_prediction_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no prediction"):
        pipeline.find_prediction(tmp_path, "LC80000000000000LGN00")