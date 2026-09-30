"""Tabular views over confusion matrices."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from cloudband.eval.confusion import BinaryConfusion
from cloudband.eval.metrics import (
    all_metrics,
    balanced_overall_accuracy,
    producer_accuracy,
    user_accuracy,
)

# The three experiments of Aybar et al. (2022), Table 6.
CLOUDSEN12_EXPERIMENTS: tuple[str, ...] = (
    "cloud/no cloud",
    "cloud shadow",
    "valid/invalid",
)

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

    Superseded by cloudsen12_table, which also masks UA in scenes without
    cloud and takes invalid as the positive class of valid/invalid; both
    were checked against the authors' per-patch metrics. Kept so existing
    callers keep working.

    Columns: median_boa, pa_low/pa_middle/pa_high, ua_low/ua_middle/ua_high,
    n_patches — the column layout of the convention Aybar et al. (2022)
    describe for CloudSEN12, so results can sit next to tables reported that
    way without reshaping. This convention is not the OmniCloudMask paper's,
    which reports PixBox and PlanetScope tables only. median_boa is a
    percentage (0-100), matching the pa_/ua_ bucket columns.
    n_patches counts scenes with a defined BOA, the ones the median uses.

    The pa_ and ua_ columns of the "clear" row take clear as the positive
    class. The CloudSEN12 valid/invalid experiment takes INVALID (cloud or
    shadow) as the positive class, so those two rows are not the same
    quantity; median_boa and n_patches are identical either way, because BOA
    is symmetric in the two classes. The paper's own PA percentages only fit
    denominators of 777, 676 and 777 IPs (cloud, shadow, valid/invalid), that
    is, IPs where the positive class exists, which is what dropna does here.
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


def swap_positive(confusion: BinaryConfusion, label: str) -> BinaryConfusion:
    """The same counts with the positive and negative classes exchanged."""
    return BinaryConfusion(
        label, tp=confusion.tn, tn=confusion.tp, fp=confusion.fn, fn=confusion.fp
    )


def cloudsen12_table(
    per_scene: dict[str, dict[str, BinaryConfusion]], decimals: int = 2
) -> pd.DataFrame:
    """The CloudSEN12 reporting convention (Aybar et al. 2022, Table 6).

    per_scene maps each scene to its clear, cloud and shadow confusions, as
    RunResult.per_scene does. Verified against the authors' per-patch metrics,
    where these rules reproduce all 22 published rows:

    - PA is NaN in a scene where the class is absent from the reference.
    - UA is NaN in every experiment for a scene with no thick or thin cloud in
      the reference, and also where nothing was predicted as the class.
    - BOA is the median over scenes where it is defined, meaning both the class
      and its complement are present.
    - The PA and UA percentages are shares of the scenes where that metric is
      defined, in three ranges: below 0.1, 0.1 to 0.9 inclusive, above 0.9.
    - valid/invalid takes INVALID (thick cloud, thin cloud, shadow) as the
      positive class. It is the clear confusion with the two classes swapped,
      so its BOA equals that of clear.

    Columns: median_boa (percent), pa_low/middle/high and ua_low/middle/high
    (percent), n_patches (scenes with a defined BOA), n_pa and n_ua (the
    denominators of the PA and UA percentages).
    """
    boa_rows, pa_rows, ua_rows = {}, {}, {}
    for scene, confusions in per_scene.items():
        no_cloud = confusions["cloud"].actual_positive == 0
        by_experiment = {
            "cloud/no cloud": confusions["cloud"],
            "cloud shadow": confusions["shadow"],
            "valid/invalid": swap_positive(confusions["clear"], "invalid"),
        }
        boa_rows[scene] = {
            n: balanced_overall_accuracy(c) for n, c in by_experiment.items()
        }
        pa_rows[scene] = {n: producer_accuracy(c) for n, c in by_experiment.items()}
        ua_rows[scene] = {
            n: float("nan") if no_cloud else user_accuracy(c)
            for n, c in by_experiment.items()
        }

    boa = pd.DataFrame.from_dict(boa_rows, orient="index")
    pa = pd.DataFrame.from_dict(pa_rows, orient="index")
    ua = pd.DataFrame.from_dict(ua_rows, orient="index")

    table = pd.DataFrame({"median_boa": (median_metric(boa) * 100).round(decimals)})
    table = table.join(bucket_percentages(pa).add_prefix("pa_").round(decimals))
    table = table.join(bucket_percentages(ua).add_prefix("ua_").round(decimals))
    table["n_patches"] = boa.notna().sum()
    table["n_pa"] = pa.notna().sum()
    table["n_ua"] = ua.notna().sum()
    table.index.name = "experiment"
    return table.loc[list(CLOUDSEN12_EXPERIMENTS)]