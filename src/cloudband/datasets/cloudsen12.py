"""Access to the CloudSEN12+ collection through tacoreader.

Two loading modes are supported. Remote loading streams from the cloud-optimised
dataset. Local loading reads previously fetched .taco parts, which is what a
Colab session should use once the parts sit on Drive.

Column names are validated against the loaded table rather than assumed, so a
schema change fails immediately with the available columns listed.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray

REMOTE_L1C = "tacofoundation:cloudsen12-l1c"
REMOTE_L2A = "tacofoundation:cloudsen12-l2a"
REMOTE_EXTRA = "tacofoundation:cloudsen12-extra"

ID_COLUMN = "tortilla:id"
SPLIT_COLUMN = "tortilla:data_split"
LABEL_TYPE_COLUMN = "label_type"
PATCH_SHAPE_COLUMN = "real_proj_shape"
ROI_COLUMN = "roi_id"
SCENE_COLUMN = "s2_id"

CLASS_PERCENTAGE_COLUMNS: tuple[str, ...] = (
    "clear_percentage",
    "thick_percentage",
    "thin_percentage",
    "cloud_shadow_percentage",
)

HIGH_QUALITY = "high"
PATCH_509 = 509
PATCH_2000 = 2000

# Reserved value marking absent data in both the reflectance stack and the
# annotation. Scribble and nolabel patches carry it inside the annotation.
NODATA_VALUE = 99

# Documented split sizes for the high quality p509 subset. Release 1.1.0 removed
# two patches for poor quality, so an observed shortfall of one or two per split
# is expected rather than an error.
EXPECTED_SPLIT_SIZES: dict[str, int] = {"train": 8490, "validation": 535, "test": 975}
REMOVED_PATCH_ALLOWANCE = 2

# Patches are padded from 509 to 512 with zeros on the left and bottom sides.
CANVAS_SIZE = 512
VALID_SIZE = 509
VALID_ROWS = slice(0, 509)
VALID_COLUMNS = slice(3, 512)

IMAGE_INDEX = 0
ANNOTATION_INDEX = 1

ScalarT = TypeVar("ScalarT", bound=np.generic)


@dataclass(frozen=True)
class Sample:
    """One patch: the reflectance stack, the annotation, and its identifiers."""

    identifier: str
    image: NDArray[np.integer]
    annotation: NDArray[np.integer]
    roi_id: str | None = None


def load_remote(name: str = REMOTE_L1C) -> pd.DataFrame:
    """Load the dataset index without downloading the archives."""
    import tacoreader

    table: pd.DataFrame = tacoreader.load(name)
    return table


def load_local(parts: Sequence[Path]) -> pd.DataFrame:
    """Load and concatenate previously downloaded .taco parts."""
    missing = [str(path) for path in parts if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing taco parts: {missing}")

    import tacoreader

    frames = [tacoreader.load(str(path)) for path in parts]
    table: pd.DataFrame = pd.concat(frames, ignore_index=True)
    return table


def require_columns(table: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise when a column the pipeline depends on is absent."""
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"missing columns {missing}; found {sorted(table.columns)}")


def select(
    table: pd.DataFrame,
    split: str | None = None,
    label_type: str | None = HIGH_QUALITY,
    patch_shape: int | None = PATCH_509,
) -> pd.DataFrame:
    """Filter the index by split, annotation quality and patch size.

    The p509 and p2000 subsets share the index, so patch size must be filtered
    explicitly or the two get mixed.
    """
    selected = table
    if split is not None:
        require_columns(table, [SPLIT_COLUMN])
        selected = selected[selected[SPLIT_COLUMN] == split]
    if label_type is not None:
        require_columns(table, [LABEL_TYPE_COLUMN])
        selected = selected[selected[LABEL_TYPE_COLUMN] == label_type]
    if patch_shape is not None:
        require_columns(table, [PATCH_SHAPE_COLUMN])
        selected = selected[selected[PATCH_SHAPE_COLUMN] == patch_shape]
    if selected.empty:
        raise ValueError(
            f"no rows for split={split!r} label_type={label_type!r} patch_shape={patch_shape!r}"
        )
    return selected.reset_index(drop=True)


def verify_split_sizes(
    table: pd.DataFrame,
    label_type: str | None = HIGH_QUALITY,
    patch_shape: int | None = PATCH_509,
) -> dict[str, int]:
    """Compare observed split sizes against the documented ones.

    Returns observed minus expected for splits that differ by more than the
    allowance for patches withdrawn in release 1.1.0.
    """
    differences = {}
    for split, expected in EXPECTED_SPLIT_SIZES.items():
        try:
            observed = len(select(table, split, label_type, patch_shape))
        except ValueError:
            observed = 0
        delta = observed - expected
        if not -REMOVED_PATCH_ALLOWANCE <= delta <= 0:
            differences[split] = delta
    return differences


def expected_pairable_scenes(table: pd.DataFrame) -> dict[str, int]:
    """Count scenes where each evaluation class is present in the annotation.

    Balanced accuracy needs both positives and negatives, so a scene where a
    class covers none or all of the patch yields no defined value for it. The
    annotator percentages let that be counted before any inference runs.
    """
    require_columns(table, CLASS_PERCENTAGE_COLUMNS)
    clear = table["clear_percentage"]
    cloud = table["thick_percentage"] + table["thin_percentage"]
    shadow = table["cloud_shadow_percentage"]
    return {
        "clear": int(((clear > 0) & (clear < 100)).sum()),
        "cloud": int(((cloud > 0) & (cloud < 100)).sum()),
        "shadow": int(((shadow > 0) & (shadow < 100)).sum()),
    }


def valid_window(array: NDArray[ScalarT]) -> NDArray[ScalarT]:
    """Crop a padded 512 canvas down to the 509 region holding real data."""
    if array.shape[-2:] != (CANVAS_SIZE, CANVAS_SIZE):
        raise ValueError(f"expected a {CANVAS_SIZE}x{CANVAS_SIZE} canvas, got {array.shape}")
    cropped: NDArray[ScalarT] = array[..., VALID_ROWS, VALID_COLUMNS]
    return cropped


def nodata_mask(annotation: NDArray[np.integer]) -> NDArray[np.bool_]:
    """Mark pixels carrying the reserved no-data value."""
    marked: NDArray[np.bool_] = annotation == NODATA_VALUE
    return marked


def read_sample(table: pd.DataFrame, index: int, crop_to_valid: bool = True) -> Sample:
    """Read one patch and its annotation from the index."""
    import rasterio

    require_columns(table, [ID_COLUMN])
    row = table.read(index)
    metadata = table.iloc[index]

    with rasterio.open(row.read(IMAGE_INDEX)) as source:
        image: NDArray[np.integer] = source.read()
    with rasterio.open(row.read(ANNOTATION_INDEX)) as source:
        annotation: NDArray[np.integer] = source.read(1)

    if crop_to_valid:
        image = valid_window(image)
        annotation = valid_window(annotation)
    return Sample(
        identifier=str(metadata[ID_COLUMN]),
        image=image,
        annotation=annotation,
        roi_id=str(metadata[ROI_COLUMN]) if ROI_COLUMN in table.columns else None,
    )


def iter_samples(
    table: pd.DataFrame, crop_to_valid: bool = True, limit: int | None = None
) -> Iterator[Sample]:
    """Yield samples in index order, optionally stopping early."""
    count = len(table) if limit is None else min(limit, len(table))
    for index in range(count):
        yield read_sample(table, index, crop_to_valid)