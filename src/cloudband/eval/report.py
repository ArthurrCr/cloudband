"""Tabular views over confusion matrices."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from cloudband.eval.confusion import BinaryConfusion
from cloudband.eval.metrics import all_metrics

COUNT_COLUMNS = ("tp", "tn", "fp", "fn")
METRIC_COLUMNS = ("ua", "pa", "oa", "boa", "f1", "iou")

# The CloudSEN12 paper's own convention (Aybar et al. 2022, "CloudSEN12, a
# global dataset..."): "we consider the following three scenarios: (i) low
# values group, which represents the percentage of IPs with PA/UA values
# lower than 0.1; (ii) middle values group ... between 0.1 and 0.9; (iii)
# high values group ... higher than 0.9." Each bucket is the percentage of
# SCENES landing in that range, not a percentile of the metric's own
# distribution across scenes — a different statistic with a similar name.
PA_UA_BUCKETS: tuple[tuple[str, Callable[[pd.Series], pd.Series]], ...] = (
    ("low", lambda values: values < 0.1),
    ("middle", lambda values: (values >= 0.1) & (values <= 0.9)),
    ("high", lambda values: values > 0.9),
)


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


def median_metric(per_scene: pd.DataFrame) -> pd.Series:
    """Median of a per-scene metric across scenes, one value per experiment.

    Matches the CloudSEN12 paper's own Median BOA convention: unlike PA and
    UA, which the paper buckets into low/middle/high groups (see
    PA_UA_BUCKETS), BOA is reported as a single median value per experiment.
    per_scene holds fractions in [0, 1], matching per_scene_metric's own
    output; the median is returned on that same scale, not as a percentage.
    """
    return per_scene.median()


def bucket_percentages(
    per_scene: pd.DataFrame,
    buckets: tuple[tuple[str, Callable[[pd.Series], pd.Series]], ...] = PA_UA_BUCKETS,
) -> pd.DataFrame:
    """Percentage of scenes whose metric value falls in each bucket, per experiment.

    Matches the CloudSEN12 paper's PAlow/PAmiddle/PAhigh convention: the
    percentage of scenes (patches) landing in each bucket, not a percentile
    of the metric's own distribution. A scene with no defined value for an
    experiment (the class absent from that scene's reference) is excluded
    from that experiment's total, not counted as a failure.
    """
    rows = []
    for experiment in per_scene.columns:
        values = per_scene[experiment].dropna()
        total = len(values)
        row: dict[str, object] = {"experiment": experiment}
        for label, predicate in buckets:
            in_bucket = predicate(values).sum()
            row[label] = (100.0 * in_bucket / total) if total else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).set_index("experiment")


def paper_style_table(
    per_scene_boa: pd.DataFrame,
    per_scene_pa: pd.DataFrame,
    per_scene_ua: pd.DataFrame,
    n_patches: dict[str, int],
    decimals: int = 2,
) -> pd.DataFrame:
    """Assemble the CloudSEN12 paper's own reporting table for one run.

    Columns: median_boa, pa_low/pa_middle/pa_high, ua_low/ua_middle/ua_high,
    n_patches — matching Aybar et al. (2022) Table 6's layout, so a result
    from this project's own evaluation can sit next to the paper's published
    numbers without reshaping. median_boa is a percentage (0-100), matching
    the pa_/ua_ bucket columns, which are already percentages.
    """
    median_boa = (median_metric(per_scene_boa) * 100).round(decimals)
    table = pd.DataFrame({"median_boa": median_boa})
    pa_buckets = bucket_percentages(per_scene_pa).add_prefix("pa_").round(decimals)
    ua_buckets = bucket_percentages(per_scene_ua).add_prefix("ua_").round(decimals)
    table = table.join(pa_buckets)
    table = table.join(ua_buckets)
    table["n_patches"] = pd.Series(n_patches)
    table.index.name = "experiment"
    return table