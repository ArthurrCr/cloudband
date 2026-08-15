"""Accuracy metrics derived from a binary one-vs-rest confusion matrix.

All metrics return a fraction in [0, 1], or nan when the denominator is zero.
"""

from __future__ import annotations

import math

from cloudband.eval.confusion import BinaryConfusion


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return math.nan
    return numerator / denominator


def producer_accuracy(confusion: BinaryConfusion) -> float:
    """Sensitivity: share of reference positives that were predicted positive."""
    return _ratio(confusion.tp, confusion.actual_positive)


def user_accuracy(confusion: BinaryConfusion) -> float:
    """Precision: share of predicted positives that are reference positives."""
    return _ratio(confusion.tp, confusion.tp + confusion.fp)


def specificity(confusion: BinaryConfusion) -> float:
    """Share of reference negatives that were predicted negative."""
    return _ratio(confusion.tn, confusion.actual_negative)


def overall_accuracy(confusion: BinaryConfusion) -> float:
    return _ratio(confusion.tp + confusion.tn, confusion.total)


def balanced_overall_accuracy(confusion: BinaryConfusion) -> float:
    """Mean of sensitivity and specificity."""
    sensitivity = producer_accuracy(confusion)
    if math.isnan(sensitivity):
        return math.nan
    negative_rate = specificity(confusion)
    if math.isnan(negative_rate):
        return math.nan
    return 0.5 * (sensitivity + negative_rate)


def f1_score(confusion: BinaryConfusion) -> float:
    precision = user_accuracy(confusion)
    recall = producer_accuracy(confusion)
    if math.isnan(precision) or math.isnan(recall) or precision + recall == 0:
        return math.nan
    return 2 * precision * recall / (precision + recall)


def intersection_over_union(confusion: BinaryConfusion) -> float:
    return _ratio(confusion.tp, confusion.tp + confusion.fp + confusion.fn)


def all_metrics(confusion: BinaryConfusion) -> dict[str, float]:
    """Return every metric keyed by name."""
    return {
        "pa": producer_accuracy(confusion),
        "ua": user_accuracy(confusion),
        "oa": overall_accuracy(confusion),
        "boa": balanced_overall_accuracy(confusion),
        "f1": f1_score(confusion),
        "iou": intersection_over_union(confusion),
    }