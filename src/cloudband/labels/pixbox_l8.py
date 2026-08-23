"""PixBox Landsat 8 reference labels.

The schema differs from the Sentinel-2 collection. Cloud presence lives in
PIXEL_SURFACE_TYPE_ID rather than CLOUD_CHARACTERISTICS_ID, which here carries
cloud type and is unrelated to presence. Shadow is a plain flag with no
distinction between topographic and cloud shadow, unlike the Sentinel-2 table.

Surface type and shadow are independent attributes, so a pixel can be labelled
both cloud and cloud shadow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

SURFACE_COLUMN = "PIXEL_SURFACE_TYPE_ID"
SHADOW_COLUMN = "CLOUD_SHADOW_ID"
PRODUCT_COLUMN = "PRODUCT_ID"
ROW_COLUMN = "PIXEL_Y"
COLUMN_COLUMN = "PIXEL_X"

# CLOUD_CHARACTERISTICS_ID exists in this collection but encodes cloud type,
# not presence. Reading it as the Sentinel-2 column of the same name would map
# only the cirrus pixels as cloud.
CLOUD_TYPE_COLUMN = "CLOUD_CHARACTERISTICS_ID"

SURFACE_CLOUD_IDS = frozenset({0, 1})
SURFACE_NOT_CLOUD_IDS = frozenset({2, 3, 4, 5, 6, 7, 11, 12})
SURFACE_MIXED_CLOUD_IDS = frozenset({8, 9, 10})
SURFACE_OTHER_IDS = frozenset({13})

SHADOW_IDS = frozenset({1})
NOT_SHADOW_IDS = frozenset({0})

TOTAL_PIXELS = 18830

EXPECTED_LABEL_COUNTS: dict[str, int] = {
    "clear": 12365,
    "cloud": 5478,
    "shadow": 1396,
}

EXPECTED_CLOUD_SHADOW_OVERLAP = 409

PRODUCT_ID_TO_SCENE: dict[int, str] = {
    1093075446: "LC81970222014109LGN00",
    1113395679: "LC81980232014276LGN00",
    1113398638: "LC82030242015058LGN00",
    1191608737: "LC81960302014022LGN00",
    1278115010: "LC81970182015080LGN00",
    1278117898: "LC81970222013186LGN00",
    1710711454: "LC81980222014260LGN00",
    1797484437: "LC81990242014075LGN00",
    1978132186: "LC81990242014107LGN00",
    2055810806: "LC82030242014103LGN00",
    2055813758: "LC82040212013251LGN00",
}


def _mask_from_ids(values: pd.Series, ids: frozenset[int]) -> NDArray[np.bool_]:
    mask: NDArray[np.bool_] = values.isin(ids).to_numpy(dtype=bool)
    return mask


def require_columns(table: pd.DataFrame) -> None:
    """Raise when a column the pipeline depends on is absent."""
    required = (SURFACE_COLUMN, SHADOW_COLUMN, PRODUCT_COLUMN, ROW_COLUMN, COLUMN_COLUMN)
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"missing columns {missing}; found {sorted(table.columns)}")


def label_masks(table: pd.DataFrame) -> dict[str, NDArray[np.bool_]]:
    """Return one independent boolean reference mask per evaluation class."""
    require_columns(table)
    surface = table[SURFACE_COLUMN]
    known = SURFACE_CLOUD_IDS | SURFACE_NOT_CLOUD_IDS | SURFACE_MIXED_CLOUD_IDS | SURFACE_OTHER_IDS
    unknown = sorted(set(surface.unique()) - known)
    if unknown:
        raise ValueError(f"unmapped values in {SURFACE_COLUMN}: {unknown}")

    cloud = _mask_from_ids(surface, SURFACE_CLOUD_IDS)
    not_cloud = _mask_from_ids(surface, SURFACE_NOT_CLOUD_IDS)
    shadow = _mask_from_ids(table[SHADOW_COLUMN], SHADOW_IDS)
    not_shadow = _mask_from_ids(table[SHADOW_COLUMN], NOT_SHADOW_IDS)
    return {"clear": not_cloud & not_shadow, "cloud": cloud, "shadow": shadow}


def label_counts(masks: dict[str, NDArray[np.bool_]]) -> dict[str, int]:
    return {name: int(np.count_nonzero(mask)) for name, mask in masks.items()}


def cloud_shadow_overlap(masks: dict[str, NDArray[np.bool_]]) -> int:
    """Count pixels labelled both cloud and cloud shadow."""
    return int(np.count_nonzero(masks["cloud"] & masks["shadow"]))


def verify_label_counts(table: pd.DataFrame) -> dict[str, int]:
    """Compare observed label counts against the reference collection."""
    masks = label_masks(table)
    observed = label_counts(masks)
    differences = {
        name: observed[name] - expected
        for name, expected in EXPECTED_LABEL_COUNTS.items()
        if observed[name] != expected
    }
    overlap = cloud_shadow_overlap(masks)
    if overlap != EXPECTED_CLOUD_SHADOW_OVERLAP:
        differences["overlap"] = overlap - EXPECTED_CLOUD_SHADOW_OVERLAP
    if len(table) != TOTAL_PIXELS:
        differences["total"] = len(table) - TOTAL_PIXELS
    return differences


def sample_predictions(table: pd.DataFrame, prediction: NDArray[np.integer]) -> NDArray[np.integer]:
    """Read predicted classes at the pixel indices carried by the table."""
    require_columns(table)
    rows = table[ROW_COLUMN].to_numpy()
    columns = table[COLUMN_COLUMN].to_numpy()
    height, width = prediction.shape
    out_of_bounds = (rows >= height) | (columns >= width) | (rows < 0) | (columns < 0)
    if out_of_bounds.any():
        raise ValueError(
            f"{int(out_of_bounds.sum())} pixel indices fall outside the "
            f"{height}x{width} prediction raster"
        )
    sampled: NDArray[np.integer] = prediction[rows, columns]
    return sampled