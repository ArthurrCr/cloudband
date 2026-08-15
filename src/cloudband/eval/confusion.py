"""Binary confusion matrices for independent per-class evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BinaryConfusion:
    """Counts for one class evaluated against everything that is not that class."""

    label: str
    tp: int
    tn: int
    fp: int
    fn: int

    def __post_init__(self) -> None:
        for name in ("tp", "tn", "fp", "fn"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def total(self) -> int:
        return self.tp + self.tn + self.fp + self.fn

    @property
    def actual_positive(self) -> int:
        return self.tp + self.fn

    @property
    def actual_negative(self) -> int:
        return self.tn + self.fp


def confusion_from_masks(
    label: str,
    reference: NDArray[np.bool_],
    prediction: NDArray[np.bool_],
    valid_mask: NDArray[np.bool_] | None = None,
) -> BinaryConfusion:
    """Build a confusion matrix from two boolean masks of equal shape.

    Elements where valid_mask is False are excluded from all counts.
    """
    if reference.shape != prediction.shape:
        raise ValueError(
            f"shape mismatch: reference {reference.shape}, prediction {prediction.shape}"
        )

    reference_flat = reference.reshape(-1)
    prediction_flat = prediction.reshape(-1)
    if valid_mask is not None:
        if valid_mask.shape != reference.shape:
            raise ValueError("valid_mask shape must match reference shape")
        keep = valid_mask.reshape(-1)
        reference_flat = reference_flat[keep]
        prediction_flat = prediction_flat[keep]

    return BinaryConfusion(
        label=label,
        tp=int(np.count_nonzero(reference_flat & prediction_flat)),
        tn=int(np.count_nonzero(~reference_flat & ~prediction_flat)),
        fp=int(np.count_nonzero(~reference_flat & prediction_flat)),
        fn=int(np.count_nonzero(reference_flat & ~prediction_flat)),
    )


def confusion_from_labels(
    label: str,
    reference: NDArray[np.integer],
    prediction: NDArray[np.integer],
    positive_class: int,
    valid_mask: NDArray[np.bool_] | None = None,
) -> BinaryConfusion:
    """Build a confusion matrix from exclusive integer label arrays.

    Use this for densely labelled rasters where every pixel holds exactly one
    class. Use confusion_from_masks when classes may overlap.
    """
    return confusion_from_masks(
        label,
        reference == positive_class,
        prediction == positive_class,
        valid_mask,
    )


def add(left: BinaryConfusion, right: BinaryConfusion) -> BinaryConfusion:
    """Sum two confusion matrices carrying the same label."""
    if left.label != right.label:
        raise ValueError(f"cannot add confusion matrices for {left.label} and {right.label}")
    return BinaryConfusion(
        label=left.label,
        tp=left.tp + right.tp,
        tn=left.tn + right.tn,
        fp=left.fp + right.fp,
        fn=left.fn + right.fn,
    )