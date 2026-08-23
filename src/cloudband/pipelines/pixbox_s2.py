"""End to end scoring of a prediction set against the PixBox Sentinel-2 labels.

The notebook calls these functions; no logic belongs in the notebook itself.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from numpy.typing import NDArray

from cloudband.eval.confusion import BinaryConfusion
from cloudband.eval.experiments import prediction_masks, run_experiments
from cloudband.labels import pixbox_s2

PREDICTION_COLUMN = "predicted_class"

# A Sentinel-2 L1C scene at 10 m, which the label pixel indices address.
EXPECTED_SCENE_SIZE = 10980


def load_reference(
    csv_path: Path, verify: bool = True, drop_unavailable: bool = True
) -> pd.DataFrame:
    """Read the label table, check it, and drop scenes whose imagery is absent.

    Verification runs against the full collection before any scene is dropped,
    so a corrupted file is still caught.
    """
    table = pd.read_csv(csv_path)
    pixbox_s2.require_columns(table)
    if verify:
        differences = pixbox_s2.verify_label_counts(table)
        if differences:
            raise ValueError(f"label counts differ from the reference collection: {differences}")
    if drop_unavailable:
        table = pixbox_s2.drop_unavailable_scenes(table)
    return table


def find_prediction(predictions_dir: Path, scene_name: str) -> Path:
    """Locate the single prediction raster belonging to a scene."""
    matches = sorted(predictions_dir.glob(f"{scene_name}*.tif"))
    if not matches:
        raise FileNotFoundError(f"no prediction for {scene_name} in {predictions_dir}")
    if len(matches) > 1:
        raise ValueError(f"{len(matches)} predictions match {scene_name}: {matches}")
    return matches[0]


def read_prediction(
    path: Path, expected_size: int | None = EXPECTED_SCENE_SIZE
) -> NDArray[np.integer]:
    """Read the first band, checking the grid the label indices assume."""
    with rasterio.open(path) as source:
        array: NDArray[np.integer] = source.read(1)
    if expected_size is not None and array.shape != (expected_size, expected_size):
        raise ValueError(
            f"{path.name} is {array.shape}, expected "
            f"({expected_size}, {expected_size}); label pixel indices assume that grid"
        )
    return array


def attach_predictions(
    table: pd.DataFrame,
    predictions_dir: Path,
    expected_size: int | None = EXPECTED_SCENE_SIZE,
) -> pd.DataFrame:
    """Sample predictions at every labelled pixel, scene by scene."""
    missing = sorted(set(table[pixbox_s2.PRODUCT_COLUMN]) - set(pixbox_s2.PRODUCT_ID_TO_SCENE))
    if missing:
        raise ValueError(f"no scene name known for product ids {missing}")

    parts = []
    for product_id, group in table.groupby(pixbox_s2.PRODUCT_COLUMN, sort=True):
        scene_name = pixbox_s2.PRODUCT_ID_TO_SCENE[int(product_id)]
        prediction = read_prediction(find_prediction(predictions_dir, scene_name), expected_size)
        sampled = group.copy()
        sampled[PREDICTION_COLUMN] = pixbox_s2.sample_predictions(group, prediction)
        parts.append(sampled)
    return pd.concat(parts).loc[table.index]


def score(table: pd.DataFrame) -> dict[str, BinaryConfusion]:
    """Build one confusion matrix per experiment over the whole collection."""
    if PREDICTION_COLUMN not in table.columns:
        raise ValueError(f"missing {PREDICTION_COLUMN}; call attach_predictions first")
    reference = pixbox_s2.label_masks(table)
    predicted = prediction_masks(table[PREDICTION_COLUMN].to_numpy())
    return run_experiments(reference, predicted)


def score_per_scene(table: pd.DataFrame) -> dict[str, dict[str, BinaryConfusion]]:
    """Confusion matrices per scene, keyed by scene name then experiment.

    Per scene counts are what a paired test consumes; the pooled result is
    recovered with eval.experiments.pool.
    """
    per_scene = {}
    for product_id, group in table.groupby(pixbox_s2.PRODUCT_COLUMN, sort=True):
        scene_name = pixbox_s2.PRODUCT_ID_TO_SCENE[int(product_id)]
        per_scene[scene_name] = score(group)
    return per_scene