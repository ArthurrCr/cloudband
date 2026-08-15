"""Contract tests for the accuracy metrics.

The published cases pin the metric definitions against externally reported
confusion matrices. A failure here means the metric implementation drifted, not
that a model got worse.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score

from cloudband.eval.confusion import BinaryConfusion, confusion_from_labels, confusion_from_masks
from cloudband.eval.metrics import balanced_overall_accuracy, overall_accuracy

# source, class, tp, tn, fp, fn, total, reported boa
PUBLISHED_CASES: tuple[tuple[str, str, int, int, int, int, int, float], ...] = (
    # OCM as reported in the paper, PixBox Sentinel-2
    ("paper_ocm_l1c", "clear", 7791, 8186, 868, 506, 17351, 92.2),
    ("paper_ocm_l1c", "cloud", 7170, 8683, 499, 999, 17351, 91.2),
    ("paper_ocm_l1c", "shadow", 778, 15860, 245, 468, 17351, 80.5),
    # OCM 1.7.0 as recomputed by the reference notebook, same dataset
    ("ocm_1_7_0", "clear", 7837, 8184, 870, 460, 17351, 92.42),
    ("ocm_1_7_0", "cloud", 7185, 8731, 451, 984, 17351, 91.52),
    ("ocm_1_7_0", "shadow", 798, 15895, 210, 448, 17351, 81.37),
    # PixBox Landsat 8
    ("paper_ocm", "clear_l8", 12290, 5399, 1066, 75, 18830, 91.5),
    ("paper_ocm", "cloud_l8", 4586, 13247, 105, 892, 18830, 91.5),
    ("paper_ocm", "shadow_l8", 710, 17361, 73, 686, 18830, 75.2),
    # Comparison methods, PixBox Sentinel-2
    ("clouds2mask", "clear", 7660, 8233, 821, 637, 17351, 91.6),
    ("clouds2mask", "cloud", 7323, 8531, 651, 846, 17351, 91.3),
    ("clouds2mask", "shadow", 721, 15930, 175, 525, 17351, 78.4),
    ("unetmobv2", "clear", 7734, 8011, 1043, 563, 17351, 90.8),
    ("unetmobv2", "cloud", 7086, 8642, 540, 1083, 17351, 90.4),
    ("unetmobv2", "shadow", 734, 15891, 214, 512, 17351, 78.8),
    ("fmask46", "clear_l8", 11564, 4426, 2039, 801, 18830, 81.0),
    ("fmask46", "cloud_l8", 3832, 12744, 608, 1646, 18830, 82.7),
    ("fmask46", "shadow_l8", 444, 17091, 343, 952, 18830, 64.9),
)


@pytest.mark.parametrize("case", PUBLISHED_CASES, ids=lambda case: f"{case[0]}_{case[1]}")
def test_boa_matches_reported_value(
    case: tuple[str, str, int, int, int, int, int, float],
) -> None:
    _, label, tp, tn, fp, fn, total, reported = case
    confusion = BinaryConfusion(label=label, tp=tp, tn=tn, fp=fp, fn=fn)
    assert confusion.total == total
    assert balanced_overall_accuracy(confusion) * 100 == pytest.approx(reported, abs=0.05)


def test_boa_equals_sklearn_balanced_accuracy() -> None:
    rng = np.random.default_rng(0)
    reference = rng.integers(0, 3, size=5000)
    prediction = rng.integers(0, 3, size=5000)
    for positive_class in (0, 1, 2):
        confusion = confusion_from_labels("c", reference, prediction, positive_class)
        expected = balanced_accuracy_score(
            reference == positive_class, prediction == positive_class
        )
        assert balanced_overall_accuracy(confusion) == pytest.approx(expected)


def test_overall_accuracy_uses_true_positives_and_negatives() -> None:
    confusion = BinaryConfusion(label="c", tp=3, tn=5, fp=1, fn=1)
    assert overall_accuracy(confusion) == pytest.approx(8 / 10)


def test_perfect_and_inverted_predictions() -> None:
    reference = np.array([False, True, True, False])
    assert balanced_overall_accuracy(confusion_from_masks("c", reference, reference)) == 1.0
    assert balanced_overall_accuracy(confusion_from_masks("c", reference, ~reference)) == 0.0


def test_boa_is_nan_when_class_absent_from_reference() -> None:
    reference = np.zeros(10, dtype=bool)
    confusion = confusion_from_masks("c", reference, reference)
    assert math.isnan(balanced_overall_accuracy(confusion))


def test_valid_mask_excludes_elements() -> None:
    reference = np.array([True, True, False, False])
    prediction = np.array([True, False, False, True])
    mask = np.array([True, True, False, False])
    confusion = confusion_from_masks("c", reference, prediction, valid_mask=mask)
    assert (confusion.tp, confusion.fn, confusion.total) == (1, 1, 2)


def test_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        confusion_from_masks("c", np.zeros(4, dtype=bool), np.zeros(5, dtype=bool))