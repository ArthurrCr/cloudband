"""Contract tests for the CloudSEN12+ labels, dataset access and scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cloudband.datasets import cloudsen12 as dataset
from cloudband.eval.confusion import BinaryConfusion
from cloudband.eval.metrics import balanced_overall_accuracy
from cloudband.labels import cloudsen12 as labels
from cloudband.pipelines import cloudsen12 as pipeline


def annotation(values: list[int], size: int = 2) -> np.ndarray:
    return np.array(values, dtype=np.int16).reshape(size, size)


def make_sample(annotation_array: np.ndarray, identifier: str = "scene_a") -> dataset.Sample:
    image = np.zeros((3, *annotation_array.shape), dtype=np.int16)
    return dataset.Sample(identifier=identifier, image=image, annotation=annotation_array)


def constant_predictor(values: list[int]) -> pipeline.Predictor:
    """Return a predictor that ignores its input and emits a fixed array."""
    fixed = annotation(values)

    def predict(image: np.ndarray) -> np.ndarray:
        del image
        return fixed

    return predict


def test_thin_cloud_folds_into_cloud() -> None:
    masks = labels.label_masks(annotation([0, 1, 2, 3]))
    assert masks["cloud"].reshape(-1).tolist() == [False, True, True, False]
    assert masks["clear"].reshape(-1).tolist() == [True, False, False, False]
    assert masks["shadow"].reshape(-1).tolist() == [False, False, False, True]


def test_classes_partition_the_array() -> None:
    array = annotation([0, 1, 2, 3])
    masks = labels.label_masks(array)
    labels.verify_partition(masks, array.size)
    assert sum(labels.label_counts(masks).values()) == array.size


def test_cloud_and_shadow_never_overlap() -> None:
    """Unlike PixBox, CloudSEN12 annotations are exclusive."""
    masks = labels.label_masks(annotation([0, 1, 2, 3]))
    assert not np.any(masks["cloud"] & masks["shadow"])


def test_unknown_annotation_code_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside the four-class scheme"):
        labels.label_masks(annotation([0, 1, 2, 7]))


def test_partition_check_catches_short_coverage() -> None:
    masks = labels.label_masks(annotation([0, 1, 2, 3]))
    with pytest.raises(ValueError, match="must partition"):
        labels.verify_partition(masks, 99)


def test_valid_window_crops_padding() -> None:
    canvas = np.zeros((3, 512, 512), dtype=np.int16)
    canvas[:, :, 0:3] = 7
    canvas[:, 509:512, :] = 7
    cropped = dataset.valid_window(canvas)
    assert cropped.shape == (3, 509, 509)
    assert not np.any(cropped == 7)


def test_valid_window_rejects_wrong_canvas() -> None:
    with pytest.raises(ValueError, match="canvas"):
        dataset.valid_window(np.zeros((3, 509, 509), dtype=np.int16))


def test_split_sizes_are_documented() -> None:
    assert dataset.EXPECTED_SPLIT_SIZES == {"train": 8490, "validation": 535, "test": 975}
    assert sum(dataset.EXPECTED_SPLIT_SIZES.values()) == 10000


def test_nodata_pixels_are_excluded_from_counts() -> None:
    """Scribble and nolabel patches carry 99 inside the annotation."""
    sample = make_sample(annotation([0, 1, 99, 3]))
    confusions = pipeline.score_sample(sample, constant_predictor([0, 1, 1, 3]))
    assert all(confusion.total == 3 for confusion in confusions.values())


def test_nodata_is_not_an_unknown_code() -> None:
    masks = labels.label_masks(annotation([0, 1, 99, 3]))
    assert not any(mask.reshape(-1)[2] for mask in masks.values())
    assert labels.valid_mask(annotation([0, 1, 99, 3])).reshape(-1).tolist() == [
        True,
        True,
        False,
        True,
    ]


def test_select_filters_patch_size() -> None:
    table = pd.DataFrame(
        {
            dataset.SPLIT_COLUMN: ["test", "test"],
            dataset.LABEL_TYPE_COLUMN: ["high", "high"],
            dataset.PATCH_SHAPE_COLUMN: [509, 2000],
        }
    )
    assert len(dataset.select(table, "test")) == 1
    assert len(dataset.select(table, "test", patch_shape=2000)) == 1


def test_removed_patches_are_tolerated() -> None:
    """Release 1.1.0 withdrew two patches, so a small shortfall is expected."""
    table = pd.DataFrame(
        {
            dataset.SPLIT_COLUMN: ["test"] * 974,
            dataset.LABEL_TYPE_COLUMN: ["high"] * 974,
            dataset.PATCH_SHAPE_COLUMN: [509] * 974,
        }
    )
    differences = dataset.verify_split_sizes(table)
    assert "test" not in differences
    assert differences["train"] == -8490


def test_expected_pairable_scenes_uses_annotator_percentages() -> None:
    table = pd.DataFrame(
        {
            "clear_percentage": [100.0, 40.0, 0.0],
            "thick_percentage": [0.0, 30.0, 60.0],
            "thin_percentage": [0.0, 30.0, 40.0],
            "cloud_shadow_percentage": [0.0, 0.0, 0.0],
        }
    )
    counts = dataset.expected_pairable_scenes(table)
    assert counts["clear"] == 1
    assert counts["cloud"] == 1
    assert counts["shadow"] == 0


def test_select_filters_and_reports_criteria() -> None:
    table = pd.DataFrame(
        {
            dataset.SPLIT_COLUMN: ["train", "test", "test"],
            dataset.LABEL_TYPE_COLUMN: ["high", "high", "scribble"],
            dataset.PATCH_SHAPE_COLUMN: [509, 509, 509],
        }
    )
    assert len(dataset.select(table, "test")) == 1
    with pytest.raises(ValueError, match="no rows for split"):
        dataset.select(table, "absent")


def test_missing_column_is_rejected() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        dataset.require_columns(pd.DataFrame({"a": [1]}), ["b"])


def test_missing_local_parts_are_reported(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="missing taco parts"):
        dataset.load_local([tmp_path / "absent.taco"])


def test_score_sample_counts_are_correct() -> None:
    sample = make_sample(annotation([0, 1, 2, 3]))
    confusions = pipeline.score_sample(sample, constant_predictor([0, 1, 1, 3]))
    assert confusions["cloud"].tp == 2
    assert confusions["clear"].tp == 1
    assert confusions["shadow"].tp == 1
    assert all(confusion.total == 4 for confusion in confusions.values())


def test_prediction_shape_mismatch_is_rejected() -> None:
    sample = make_sample(annotation([0, 1, 2, 3]))

    def wrong_shape(image: np.ndarray) -> np.ndarray:
        del image
        return np.zeros((3, 3), dtype=np.int16)

    with pytest.raises(ValueError, match="prediction is"):
        pipeline.score_sample(sample, wrong_shape)


def test_pooled_equals_whole_split() -> None:
    samples = [
        make_sample(annotation([0, 1, 2, 3]), "a"),
        make_sample(annotation([3, 3, 0, 1]), "b"),
    ]
    per_scene = pipeline.score_split(samples, constant_predictor([0, 1, 1, 3]))
    pooled = pipeline.pooled(per_scene)
    for name in ("clear", "cloud", "shadow"):
        expected = sum(per_scene[key][name].tp for key in per_scene)
        assert pooled[name].tp == expected


def test_duplicate_identifier_is_rejected() -> None:
    samples = [make_sample(annotation([0, 1, 2, 3]), "a")] * 2
    with pytest.raises(ValueError, match="duplicate scene identifier"):
        pipeline.score_split(samples, constant_predictor([0, 1, 1, 3]))


def test_empty_split_is_rejected() -> None:
    with pytest.raises(ValueError, match="no samples were scored"):
        pipeline.score_split([], constant_predictor([0, 0, 0, 0]))


def test_single_class_scene_yields_no_defined_metric() -> None:
    """A fully clear scene has no negatives, so specificity is undefined."""
    samples = [make_sample(annotation([0, 0, 0, 0]), "a")]
    per_scene = pipeline.score_split(samples, constant_predictor([0, 0, 0, 0]))
    frame = pipeline.per_scene_metric(per_scene, balanced_overall_accuracy)
    assert list(frame.columns) == ["clear", "cloud", "shadow"]
    assert frame.loc["a"].isna().all()
    assert pipeline.pairable_scene_counts(frame) == {"clear": 0, "cloud": 0, "shadow": 0}


def test_pairable_counts_differ_between_experiments() -> None:
    """Effective sample size for a paired test is smaller than the split size."""
    samples = [
        make_sample(annotation([0, 1, 1, 0]), "a"),
        make_sample(annotation([0, 0, 0, 0]), "b"),
    ]
    per_scene = pipeline.score_split(samples, constant_predictor([0, 1, 1, 0]))
    frame = pipeline.per_scene_metric(per_scene, balanced_overall_accuracy)
    counts = pipeline.pairable_scene_counts(frame)
    assert counts["cloud"] == 1
    assert counts["shadow"] == 0
    assert len(frame) == 2


def test_per_scene_metric_is_the_paired_test_input() -> None:
    samples = [
        make_sample(annotation([0, 1, 2, 3]), "a"),
        make_sample(annotation([0, 1, 1, 3]), "b"),
    ]
    per_scene = pipeline.score_split(samples, constant_predictor([0, 1, 1, 3]))
    frame = pipeline.per_scene_metric(per_scene, balanced_overall_accuracy)
    assert len(frame) == 2
    assert frame.index.tolist() == ["a", "b"]
    assert frame.loc["b", "cloud"] == 1.0


def test_confusion_label_carries_experiment_name() -> None:
    sample = make_sample(annotation([0, 1, 2, 3]))
    confusions = pipeline.score_sample(sample, constant_predictor([0, 1, 1, 3]))
    assert all(isinstance(c, BinaryConfusion) for c in confusions.values())
    assert confusions["cloud"].label == "cloud"