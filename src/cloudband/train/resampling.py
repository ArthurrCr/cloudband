"""Per-batch mixed-resolution resampling for training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import zoom

from cloudband.datasets.cloudsen12 import NODATA_VALUE
from cloudband.train.protocol import patch_size_px

BILINEAR_ORDER = 1
NEAREST_ORDER = 0
EDGE_MODE = "nearest"


def sample_gsd(min_gsd_m: float, max_gsd_m: float, rng: np.random.Generator) -> float:
    return float(rng.uniform(min_gsd_m, max_gsd_m))


def _zoom_factors(ndim: int, scale: float) -> tuple:
    return (1.0,) * (ndim - 2) + (scale, scale)


def resample_image(
    image: NDArray[np.floating],
    target_size: int,
    nodata_value: float | None = NODATA_VALUE,
) -> NDArray[np.floating]:
    """Bilinear resize, with no-data pixels re-stamped after interpolation.

    Bilinear interpolation blends the no-data sentinel into neighbouring real
    values at every boundary; re-stamping after the resize keeps the sentinel
    exact instead of leaking into nearby pixels.
    """
    scale = target_size / image.shape[-1]
    factors = _zoom_factors(image.ndim, scale)
    resized: NDArray[np.floating] = zoom(
        image, factors, order=BILINEAR_ORDER, mode=EDGE_MODE
    )
    if nodata_value is not None:
        nodata = (image == nodata_value).astype(np.uint8)
        resized_nodata = zoom(nodata, factors, order=NEAREST_ORDER, mode=EDGE_MODE) > 0
        resized[resized_nodata] = nodata_value
    return resized


def resample_annotation(
    annotation: NDArray[np.integer], target_size: int
) -> NDArray[np.integer]:
    """Nearest-neighbour resize, so no intermediate class value is invented."""
    scale = target_size / annotation.shape[-1]
    factors = _zoom_factors(annotation.ndim, scale)
    resized: NDArray[np.integer] = zoom(
        annotation, factors, order=NEAREST_ORDER, mode=EDGE_MODE
    )
    return resized


@dataclass(frozen=True)
class ResampledBatch:
    images: NDArray[np.floating]
    annotations: NDArray[np.integer]
    gsd_m: float
    size_px: int


def apply_mixed_resolution(
    images: NDArray[np.floating],
    annotations: NDArray[np.integer],
    min_gsd_m: float,
    max_gsd_m: float,
    rng: np.random.Generator,
) -> ResampledBatch:
    """Resample one batch to a single GSD drawn at random from the range."""
    gsd = sample_gsd(min_gsd_m, max_gsd_m, rng)
    size = patch_size_px(gsd)
    return ResampledBatch(
        images=resample_image(images, size),
        annotations=resample_annotation(annotations, size),
        gsd_m=gsd,
        size_px=size,
    )