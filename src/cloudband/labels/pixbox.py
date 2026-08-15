"""PixBox Sentinel-2 reference labels and scene manifest.

The collection encodes cloud and shadow as two independent integer attributes,
so a pixel can be labelled both cloud and cloud shadow. Clear is defined as the
conjunction of both negatives, not as the complement of the other two classes.

The scene manifest maps PRODUCT_ID to the Sentinel-2 L1C product name, which the
distributed table does not carry.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

CLOUD_COLUMN = "CLOUD_CHARACTERISTICS_ID"
SHADOW_COLUMN = "SHADOW_ID"
PRODUCT_COLUMN = "PRODUCT_ID"
ROW_COLUMN = "PIXEL_Y"
COLUMN_COLUMN = "PIXEL_X"

CLOUD_IDS = frozenset({2, 3, 4, 5, 6, 8, 9, 10, 11, 12})
NOT_CLOUD_IDS = frozenset({0, 1, 7})
SHADOW_IDS = frozenset({3})
NOT_SHADOW_IDS = frozenset({0, 1, 2, 4})

TOTAL_PIXELS = 17351

# Reference positives per class, used as an integrity check on the input table.
EXPECTED_LABEL_COUNTS: dict[str, int] = {
    "clear": 8297,
    "cloud": 8169,
    "shadow": 1246,
}

# Pixels labelled both cloud and cloud shadow.
EXPECTED_CLOUD_SHADOW_OVERLAP = 361

PRODUCT_ID_TO_SCENE: dict[int, str] = {
    5472572: "S2A_MSIL1C_20171205T143751_N0206_R096_T19KEU_20171205T180316",
    183807605: "S2A_MSIL1C_20170929T211511_N0205_R143_T06VWN_20170929T211510",
    273136892: "S2A_MSIL1C_20180126T182631_N0206_R127_T11SPC_20180126T201415",
    433871481: "S2A_MSIL1C_20180303T140051_N0206_R067_T19FDV_20180303T202004",
    522655885: "S2A_MSIL1C_20171107T150421_N0206_R125_T22WES_20171107T165127",
    623596858: "S2A_MSIL1C_20170629T103021_N0205_R108_T31TFJ_20170629T103020",
    701404038: "S2A_MSIL1C_20170908T100031_N0205_R122_T32SPJ_20170908T100655",
    784585929: "S2B_MSIL1C_20170731T102019_N0205_R065_T33VVE_20170731T102348",
    872013732: "S2A_MSIL1C_20180223T092031_N0206_R093_T36VUM_20180223T113049",
    885047116: "S2B_MSIL1C_20171210T060229_N0206_R091_T42RUN_20171210T071154",
    960993216: "S2A_MSIL1C_20180106T032121_N0206_R118_T48NUG_20180106T083912",
    1041598282: "S2A_MSIL1C_20180222T012651_N0206_R074_T54TWN_20180222T031349",
    1126775487: "S2B_MSIL1C_20170725T183309_N0205_R127_T11SPC_20170725T183309",
    1211409580: "S2A_MSIL1C_20170209T154541_N0204_R111_T17PPK_20170209T154543",
    1248336059: "S2A_MSIL1C_20170725T142751_N0205_R053_T19GBQ_20170725T143854",
    1309065466: "S2B_MSIL1C_20170712T113319_N0205_R080_T28PCV_20170712T114542",
    1386821964: "S2A_MSIL1C_20170726T102021_N0205_R065_T33VVE_20170726T102259",
    1407300752: "S2A_MSIL1C_20170113T072241_N0204_R006_T40UEE_20170113T072238",
    1470797653: "S2A_MSIL1C_20180217T053911_N0206_R005_T44SKJ_20180217T082149",
    1559102360: "S2A_MSIL1C_20170917T052641_N0205_R105_T48XWG_20170917T052642",
    1650478641: "S2A_MSIL1C_20170916T143741_N0205_R096_T19KEU_20170916T143942",
    1727138395: "S2A_MSIL1C_20170620T181921_N0205_R127_T11SPC_20170620T182846",
    1727140056: "S2A_MSIL1C_20180302T142851_N0206_R053_T19GBQ_20180302T192732",
    1821333386: "S2B_MSIL1C_20180302T150259_N0206_R125_T22WES_20180302T183800",
    1832501017: "S2B_MSIL1C_20170728T101029_N0205_R022_T32TPS_20170728T101024",
    1892503998: "S2B_MSIL1C_20170916T101019_N0205_R022_T32SPJ_20170916T101354",
    1899752240: "S2A_MSIL1C_20170712T071621_N0205_R006_T40UEE_20170712T071617",
    2019111565: "S2A_MSIL1C_20170706T051651_N0205_R062_T48XWG_20170706T051649",
    2065997836: "S2A_MSIL1C_20180102T140051_N0206_R067_T21LXK_20180102T154324",
}


def _mask_from_ids(values: pd.Series, ids: frozenset[int]) -> NDArray[np.bool_]:
    mask: NDArray[np.bool_] = values.isin(ids).to_numpy(dtype=bool)
    return mask


def require_columns(table: pd.DataFrame) -> None:
    """Raise when a column the pipeline depends on is absent."""
    required = (CLOUD_COLUMN, SHADOW_COLUMN, PRODUCT_COLUMN, ROW_COLUMN, COLUMN_COLUMN)
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"missing columns {missing}; found {sorted(table.columns)}")


def label_masks(table: pd.DataFrame) -> dict[str, NDArray[np.bool_]]:
    """Return one independent boolean reference mask per evaluation class.

    Cloud and shadow are not mutually exclusive. Clear requires both negatives.
    """
    require_columns(table)
    cloud = _mask_from_ids(table[CLOUD_COLUMN], CLOUD_IDS)
    not_cloud = _mask_from_ids(table[CLOUD_COLUMN], NOT_CLOUD_IDS)
    shadow = _mask_from_ids(table[SHADOW_COLUMN], SHADOW_IDS)
    not_shadow = _mask_from_ids(table[SHADOW_COLUMN], NOT_SHADOW_IDS)
    return {"clear": not_cloud & not_shadow, "cloud": cloud, "shadow": shadow}


def drop_unclassified(table: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that fall into none of the three evaluation classes."""
    masks = label_masks(table)
    keep = masks["clear"] | masks["cloud"] | masks["shadow"]
    return table.loc[keep].copy()


def label_counts(masks: dict[str, NDArray[np.bool_]]) -> dict[str, int]:
    return {name: int(np.count_nonzero(mask)) for name, mask in masks.items()}


def cloud_shadow_overlap(masks: dict[str, NDArray[np.bool_]]) -> int:
    """Count pixels labelled both cloud and cloud shadow."""
    return int(np.count_nonzero(masks["cloud"] & masks["shadow"]))


def verify_label_counts(table: pd.DataFrame) -> dict[str, int]:
    """Compare observed label counts against the expected ones.

    Returns observed minus expected for every quantity that differs. An empty
    result means the input table matches the reference collection.
    """
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
    """Read predicted classes at the pixel indices carried by the table.

    The indices address the scene raster directly, so the prediction must be on
    the native scene grid at the resolution the labels were collected at.
    """
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