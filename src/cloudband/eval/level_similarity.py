"""How different the L1C and L2A versions of one patch are, before and after normalization.

Atmospheric correction changes each band roughly by an offset and a gain, and the dynamic
Z-score takes both away. What remains after normalization is only the part of the
correction that is not affine. For one channel, with the two images normalized over the
same pixels, the root mean squared difference is sqrt(2 * (1 - r)), where r is the
Pearson correlation of the raw values: 0 when the two are the same image up to gain and
offset, about 1.41 when they are unrelated.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cloudband.datasets.cloudsen12 import ID_COLUMN, NODATA_VALUE, Sample, require_columns
from cloudband.pipelines.phase0 import RGN_BANDS
from cloudband.train.normalize import dynamic_z_score

SampleReader = Callable[[pd.DataFrame, int, bool], Sample]


def level_pairs(
    l1c: pd.DataFrame, l2a: pd.DataFrame, l2a_offset: int = 0
) -> list[tuple[int, int]]:
    """Positions of the same patch in the L1C and the L2A table, matched by patch id.

    l2a_offset is added to the L2A positions, for a table that holds the L1C rows first
    and the L2A rows after them, as the training table does.
    """
    require_columns(l1c, [ID_COLUMN])
    require_columns(l2a, [ID_COLUMN])
    where = {identifier: position for position, identifier in enumerate(l2a[ID_COLUMN])}
    return [
        (position, l2a_offset + where[identifier])
        for position, identifier in enumerate(l1c[ID_COLUMN])
        if identifier in where
    ]


def compare_channel(
    l1c: NDArray[np.number], l2a: NDArray[np.number], nodata_value: float = NODATA_VALUE
) -> dict[str, float]:
    """Raw and normalized difference between one channel of the two levels."""
    first, second = l1c.astype(np.float64), l2a.astype(np.float64)
    valid = (first != nodata_value) & (second != nodata_value)
    if valid.sum() < 2:
        return {"raw_relative_difference": np.nan, "correlation": np.nan, "z_rmse": np.nan}

    a, b = first[valid], second[valid]
    scale = np.abs(a).mean()
    raw_relative = float(np.abs(a - b).mean() / scale) if scale > 0 else np.nan

    if a.std() == 0 or b.std() == 0:
        correlation = np.nan
    else:
        correlation = float(np.corrcoef(a, b)[0, 1])

    masked_first = np.where(valid, first, nodata_value)[np.newaxis]
    masked_second = np.where(valid, second, nodata_value)[np.newaxis]
    za = dynamic_z_score(masked_first, nodata_value)[0][valid]
    zb = dynamic_z_score(masked_second, nodata_value)[0][valid]
    z_rmse = float(np.sqrt(np.mean((za - zb) ** 2)))

    return {
        "raw_relative_difference": raw_relative,
        "correlation": correlation,
        "z_rmse": z_rmse,
    }


def compare_levels(
    pairs: Iterable[tuple[int, int]],
    table: pd.DataFrame,
    read_sample: SampleReader,
    band_names: tuple[str, ...] = RGN_BANDS,
) -> pd.DataFrame:
    """One row per patch and channel, comparing its L1C and L2A versions.

    read_sample must return the bands in band_names order, as the local cache does.
    """
    rows = []
    for l1c_position, l2a_position in pairs:
        first = read_sample(table, l1c_position, True)
        second = read_sample(table, l2a_position, True)
        if first.image.shape != second.image.shape:
            raise ValueError(
                f"{first.identifier}: L1C is {first.image.shape}, L2A is {second.image.shape}"
            )
        for channel, band in enumerate(band_names):
            rows.append(
                {
                    "patch": first.identifier,
                    "band": band,
                    **compare_channel(first.image[channel], second.image[channel]),
                }
            )
    measures = ["raw_relative_difference", "correlation", "z_rmse"]
    frame = pd.DataFrame(rows, columns=["patch", "band", *measures])
    return frame.astype({measure: float for measure in measures})


def summarize(comparison: pd.DataFrame) -> pd.DataFrame:
    """Median and quartiles of each measure, per band."""
    measures = ["raw_relative_difference", "correlation", "z_rmse"]
    grouped = comparison.groupby("band", sort=False)[measures]
    summary = grouped.quantile([0.25, 0.5, 0.75]).unstack()
    summary.columns = [f"{measure} q{int(q * 100)}" for measure, q in summary.columns]
    return summary
