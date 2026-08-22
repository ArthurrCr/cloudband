"""CloudSEN12 dense annotations mapped onto the three evaluation classes.

Annotations are exclusive: every pixel carries exactly one of four classes.
Thick and thin cloud collapse into a single cloud class for evaluation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

ANNOTATION_CLEAR = 0
ANNOTATION_THICK_CLOUD = 1
ANNOTATION_THIN_CLOUD = 2
ANNOTATION_CLOUD_SHADOW = 3

ANNOTATION_CODES: dict[str, frozenset[int]] = {
    "clear": frozenset({ANNOTATION_CLEAR}),
    "cloud": frozenset({ANNOTATION_THICK_CLOUD, ANNOTATION_THIN_CLOUD}),
    "shadow": frozenset({ANNOTATION_CLOUD_SHADOW}),
}

KNOWN_CODES: frozenset[int] = frozenset().union(*ANNOTATION_CODES.values())

# Reserved value marking absent annotation, present in scribble and nolabel
# patches. High quality patches should carry none.
NODATA_VALUE = 99


def label_masks(annotation: NDArray[np.integer]) -> dict[str, NDArray[np.bool_]]:
    """Return one boolean reference mask per evaluation class.

    Unlike PixBox, the three masks partition the annotated pixels: no pixel is
    both cloud and shadow. Pixels holding the no-data value belong to no class
    and must be excluded through valid_mask.
    """
    unknown = sorted(set(np.unique(annotation).tolist()) - KNOWN_CODES - {NODATA_VALUE})
    if unknown:
        raise ValueError(f"annotation holds codes outside the four-class scheme: {unknown}")
    return {name: np.isin(annotation, list(codes)) for name, codes in ANNOTATION_CODES.items()}


def valid_mask(annotation: NDArray[np.integer]) -> NDArray[np.bool_]:
    """Mark pixels carrying an annotation, excluding the no-data value."""
    marked: NDArray[np.bool_] = annotation != NODATA_VALUE
    return marked


def label_counts(masks: dict[str, NDArray[np.bool_]]) -> dict[str, int]:
    return {name: int(np.count_nonzero(mask)) for name, mask in masks.items()}


def verify_partition(masks: dict[str, NDArray[np.bool_]], size: int) -> None:
    """Raise when the three masks fail to partition an array of the given size.

    Pass the count of annotated pixels, not the array size, when the annotation
    holds no-data. A failure means either an unmapped code slipped through or
    the class codes are not what this module assumes.
    """
    total = sum(int(np.count_nonzero(mask)) for mask in masks.values())
    if total != size:
        raise ValueError(f"masks cover {total} of {size} pixels; classes must partition")
    overlap = int(np.count_nonzero(masks["cloud"] & masks["shadow"]))
    if overlap:
        raise ValueError(f"{overlap} pixels are both cloud and shadow")