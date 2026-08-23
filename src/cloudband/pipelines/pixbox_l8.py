"""Scoring a prediction set against the PixBox Landsat 8 labels.

Landsat scenes are 30 m, outside the 10 m the models were trained on, and the
band positions differ from Sentinel-2. Both are declared here rather than shared
with the Sentinel-2 pipeline, since silently reusing either would produce
plausible wrong numbers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from numpy.typing import NDArray

from cloudband.eval.confusion import BinaryConfusion
from cloudband.eval.experiments import prediction_masks, run_experiments
from cloudband.labels import pixbox_l8 as labels

PREDICTION_COLUMN = "predicted_class"

# OLI band numbers for red, green and near-infrared, one-based as the files are
# named. Landsat NIR is B5, not B4 as on Sentinel-2.
RED_BAND = 4
GREEN_BAND = 3
NIR_BAND = 5
RGN_BANDS: tuple[int, int, int] = (RED_BAND, GREEN_BAND, NIR_BAND)

NATIVE_RESOLUTION_M = 30

# A Landsat 8 scene is not a fixed size, so no single shape can be asserted.
# The plausible range guards against a resampled or subset raster.
MIN_SCENE_SIDE = 7000
MAX_SCENE_SIDE = 9000


def load_reference(csv_path: Path, verify: bool = True) -> pd.DataFrame:
    """Read the label table and check it against the reference collection."""
    table = pd.read_csv(csv_path)
    labels.require_columns(table)
    if verify:
        differences = labels.verify_label_counts(table)
        if differences:
            raise ValueError(f"label counts differ from the reference collection: {differences}")
    return table


def find_prediction(predictions_dir: Path, scene_name: str) -> Path:
    """Locate the single prediction raster belonging to a scene."""
    matches = sorted(predictions_dir.glob(f"{scene_name}*.tif"))
    if not matches:
        raise FileNotFoundError(f"no prediction for {scene_name} in {predictions_dir}")
    if len(matches) > 1:
        raise ValueError(f"{len(matches)} predictions match {scene_name}: {matches}")
    return matches[0]


def read_prediction(path: Path, check_shape: bool = True) -> NDArray[np.integer]:
    """Read the first band, checking the raster is a full scene at 30 m.

    The label pixel indices address the native scene grid, so a resampled or
    cropped raster would sample the wrong pixels.
    """
    with rasterio.open(path) as source:
        array: NDArray[np.integer] = source.read(1)
    if check_shape:
        height, width = array.shape
        outside = not (
            MIN_SCENE_SIDE <= height <= MAX_SCENE_SIDE and MIN_SCENE_SIDE <= width <= MAX_SCENE_SIDE
        )
        if outside:
            raise ValueError(
                f"{path.name} is {array.shape}, outside the {MIN_SCENE_SIDE}-{MAX_SCENE_SIDE} "
                "range a full Landsat 8 scene occupies; label indices assume the native grid"
            )
    return array


def attach_predictions(
    table: pd.DataFrame, predictions_dir: Path, check_shape: bool = True
) -> pd.DataFrame:
    """Sample predictions at every labelled pixel, scene by scene."""
    missing = sorted(set(table[labels.PRODUCT_COLUMN]) - set(labels.PRODUCT_ID_TO_SCENE))
    if missing:
        raise ValueError(f"no scene name known for product ids {missing}")

    parts = []
    for product_id, group in table.groupby(labels.PRODUCT_COLUMN, sort=True):
        scene_name = labels.PRODUCT_ID_TO_SCENE[int(product_id)]
        prediction = read_prediction(find_prediction(predictions_dir, scene_name), check_shape)
        sampled = group.copy()
        sampled[PREDICTION_COLUMN] = labels.sample_predictions(group, prediction)
        parts.append(sampled)
    return pd.concat(parts).loc[table.index]


def score(table: pd.DataFrame) -> dict[str, BinaryConfusion]:
    """Build one confusion matrix per experiment over the whole collection."""
    if PREDICTION_COLUMN not in table.columns:
        raise ValueError(f"missing {PREDICTION_COLUMN}; call attach_predictions first")
    reference = labels.label_masks(table)
    predicted = prediction_masks(table[PREDICTION_COLUMN].to_numpy())
    return run_experiments(reference, predicted)


def score_per_scene(table: pd.DataFrame) -> dict[str, dict[str, BinaryConfusion]]:
    """Confusion matrices per scene, keyed by scene name then experiment.

    With eleven scenes a paired test has almost no power, and the collection was
    not built on a probability sampling design. These values are descriptive.
    """
    per_scene = {}
    for product_id, group in table.groupby(labels.PRODUCT_COLUMN, sort=True):
        scene_name = labels.PRODUCT_ID_TO_SCENE[int(product_id)]
        per_scene[scene_name] = score(group)
    return per_scene