"""Tabular views over confusion matrices."""

from __future__ import annotations

import pandas as pd

from cloudband.eval.confusion import BinaryConfusion
from cloudband.eval.metrics import all_metrics

COUNT_COLUMNS = ("tp", "tn", "fp", "fn")
METRIC_COLUMNS = ("ua", "pa", "oa", "boa", "f1", "iou")


def to_frame(confusions: dict[str, BinaryConfusion]) -> pd.DataFrame:
    """One row per experiment with counts and metrics as fractions."""
    rows = []
    for name, confusion in confusions.items():
        row: dict[str, object] = {"experiment": name}
        row.update({column: getattr(confusion, column) for column in COUNT_COLUMNS})
        row.update(all_metrics(confusion))
        rows.append(row)
    frame = pd.DataFrame(rows).set_index("experiment")
    return frame[[*COUNT_COLUMNS, *METRIC_COLUMNS]]


def as_percentages(frame: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    """Format the metric columns as percentages, leaving counts unchanged."""
    formatted = frame.copy()
    for column in METRIC_COLUMNS:
        if column in formatted.columns:
            formatted[column] = (formatted[column] * 100).round(decimals)
    return formatted


def compare(frames: dict[str, pd.DataFrame], column: str = "boa") -> pd.DataFrame:
    """Put one metric from several runs side by side, runs as columns."""
    series = {name: frame[column] for name, frame in frames.items()}
    return pd.DataFrame(series)