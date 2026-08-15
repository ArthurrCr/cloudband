"""The three evaluation experiments: clear, cloud and shadow.

Each experiment is an independent binary problem. Reference labels for cloud and
shadow may overlap, so the three are not a partition and cannot be derived from
a single exclusive label array.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cloudband.eval.confusion import BinaryConfusion, add, confusion_from_masks
from cloudband.eval.metrics import all_metrics

EXPERIMENTS: tuple[str, ...] = ("clear", "cloud", "shadow")

# Predicted class codes emitted by the four-class models.
PREDICTED_CLEAR = 0
PREDICTED_THICK_CLOUD = 1
PREDICTED_THIN_CLOUD = 2
PREDICTED_SHADOW = 3

PREDICTION_CODES: dict[str, frozenset[int]] = {
    "clear": frozenset({PREDICTED_CLEAR}),
    "cloud": frozenset({PREDICTED_THICK_CLOUD, PREDICTED_THIN_CLOUD}),
    "shadow": frozenset({PREDICTED_SHADOW}),
}


def prediction_masks(prediction: NDArray[np.integer]) -> dict[str, NDArray[np.bool_]]:
    """Fold predicted class codes into one boolean mask per experiment."""
    known: set[int] = set().union(*PREDICTION_CODES.values())
    unknown = sorted(set(np.unique(prediction).tolist()) - known)
    if unknown:
        raise ValueError(f"unknown predicted class codes: {unknown}")
    return {name: np.isin(prediction, list(codes)) for name, codes in PREDICTION_CODES.items()}


def run_experiments(
    reference: dict[str, NDArray[np.bool_]],
    predicted: dict[str, NDArray[np.bool_]],
    valid_mask: NDArray[np.bool_] | None = None,
) -> dict[str, BinaryConfusion]:
    """Return one confusion matrix per experiment, keyed by experiment name."""
    missing = [name for name in EXPERIMENTS if name not in reference or name not in predicted]
    if missing:
        raise ValueError(f"missing masks for experiments {missing}")
    return {
        name: confusion_from_masks(name, reference[name], predicted[name], valid_mask)
        for name in EXPERIMENTS
    }


def pool(per_unit: dict[str, dict[str, BinaryConfusion]]) -> dict[str, BinaryConfusion]:
    """Sum per-unit confusion matrices into one matrix per experiment.

    Summing counts is not the same as averaging per-unit metrics; the two differ
    whenever units hold different pixel counts.
    """
    pooled: dict[str, BinaryConfusion] = {}
    for confusions in per_unit.values():
        for name, confusion in confusions.items():
            pooled[name] = add(pooled[name], confusion) if name in pooled else confusion
    return pooled


def report(confusions: dict[str, BinaryConfusion]) -> dict[str, dict[str, float]]:
    """Return every metric for every experiment."""
    return {name: all_metrics(confusion) for name, confusion in confusions.items()}