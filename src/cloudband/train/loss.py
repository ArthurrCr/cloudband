"""Segmentation loss shared by every architecture in the comparison."""

from __future__ import annotations

from fastai.losses import CrossEntropyLossFlat

from cloudband.datasets.cloudsen12 import NODATA_VALUE

CLASS_AXIS = 1


def build_loss(ignore_index: int = NODATA_VALUE) -> CrossEntropyLossFlat:
    """Cross entropy over the class axis, no-data pixels excluded."""
    return CrossEntropyLossFlat(axis=CLASS_AXIS, ignore_index=ignore_index)