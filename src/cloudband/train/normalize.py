"""Dynamic per-sample Z-score normalization, per ADR-0023 D5."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cloudband.datasets.cloudsen12 import NODATA_VALUE


def dynamic_z_score(
    image: NDArray[np.floating], nodata_value: float = NODATA_VALUE
) -> NDArray[np.floating]:
    """Normalize each channel to zero mean, unit std, using only valid pixels.

    Mean and std are computed per input and per channel, not from a fixed
    dataset-wide constant. No-data pixels are excluded from both statistics,
    then zeroed in the output rather than left at a raw or arbitrarily large
    normalized value.
    """
    image = image.astype(np.float32)
    valid = image != nodata_value
    normalized = np.zeros_like(image)

    for channel in range(image.shape[0]):
        channel_valid = valid[channel]
        if not channel_valid.any():
            continue

        values = image[channel][channel_valid]
        mean = values.mean()
        std = values.std()
        if std == 0:
            continue

        channel_normalized = (image[channel] - mean) / std
        channel_normalized[~channel_valid] = 0.0
        normalized[channel] = channel_normalized

    return normalized