"""Scoring of a four-class model against the CloudSEN12+ annotations.

Predictions are scored per scene, because a scene is the unit a paired test
consumes. Pooling the per-scene matrices reproduces the whole-split result.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cloudband.datasets import cloudsen12 as dataset
from cloudband.eval.confusion import BinaryConfusion
from cloudband.eval.experiments import pool, prediction_masks, run_experiments
from cloudband.labels import cloudsen12 as labels

Predictor = Callable[[NDArray[np.integer]], NDArray[np.integer]]


def score_sample(
    sample: dataset.Sample,
    predictor: Predictor,
    verify: bool = True,
) -> dict[str, BinaryConfusion]:
    """Run the predictor on one patch and return one matrix per experiment.

    Pixels carrying the no-data value are excluded from every count, so a
    partially annotated patch contributes only its annotated pixels.
    """
    reference = labels.label_masks(sample.annotation)
    annotated = labels.valid_mask(sample.annotation)
    if verify:
        labels.verify_partition(reference, int(np.count_nonzero(annotated)))

    prediction = predictor(sample.image)
    if prediction.shape != sample.annotation.shape:
        raise ValueError(
            f"prediction is {prediction.shape}, annotation is {sample.annotation.shape}"
        )
    return run_experiments(reference, prediction_masks(prediction), valid_mask=annotated)


def score_split(
    samples: Iterable[dataset.Sample],
    predictor: Predictor,
    verify: bool = True,
) -> dict[str, dict[str, BinaryConfusion]]:
    """Score every sample, keyed by scene identifier then experiment."""
    per_scene: dict[str, dict[str, BinaryConfusion]] = {}
    for sample in samples:
        if sample.identifier in per_scene:
            raise ValueError(f"duplicate scene identifier {sample.identifier!r}")
        per_scene[sample.identifier] = score_sample(sample, predictor, verify)
    if not per_scene:
        raise ValueError("no samples were scored")
    return per_scene


def pooled(per_scene: dict[str, dict[str, BinaryConfusion]]) -> dict[str, BinaryConfusion]:
    """Sum the per-scene matrices into one matrix per experiment."""
    return pool(per_scene)


def per_scene_metric(
    per_scene: dict[str, dict[str, BinaryConfusion]],
    metric: Callable[[BinaryConfusion], float],
) -> pd.DataFrame:
    """One row per scene, one column per experiment.

    This is the input a paired test consumes. A cell is NaN when the class is
    absent from that scene, either as a positive or as a negative: balanced
    accuracy needs both to be defined. A fully clear scene therefore yields NaN
    for every experiment, including clear, since specificity has no denominator.
    """
    rows = {
        identifier: {name: metric(confusion) for name, confusion in confusions.items()}
        for identifier, confusions in per_scene.items()
    }
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def pairable_scene_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Count scenes with a defined value per experiment.

    A paired test runs on these counts, not on the number of scenes in the
    split. Single-class scenes drop out, so the effective sample size is smaller
    than the split size and differs between experiments.
    """
    return {column: int(frame[column].notna().sum()) for column in frame.columns}