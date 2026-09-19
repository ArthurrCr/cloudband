"""Paired comparison between two trained models' per-scene results, per note 2.10."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from scipy.stats import wilcoxon


@dataclass(frozen=True)
class PairedComparison:
    experiment: str
    n_pairs: int
    statistic: float
    p_value: float
    median_difference: float


def average_across_seeds(boa_frames: tuple) -> pd.DataFrame:
    """Average one architecture's per-scene metric across its seeds.

    Each frame is one seed's per_scene_metric output for the same
    architecture, same scenes, same experiments. Averaging first means a
    later Wilcoxon test pairs by scene, not by seed: pairing directly by
    seed caps the test at as many pairs as there are seeds, and at five
    pairs the smallest achievable two-sided p-value is 0.125 - no
    seed-level difference, however consistent, could ever reach p < 0.05.
    A scene stays undefined only if it is undefined for every seed, since
    the reason a cell is NaN (the class absent from that scene's reference
    annotation) does not depend on which seed trained the model.
    """
    if not boa_frames:
        raise ValueError("need at least one seed's results to average")

    columns = set(boa_frames[0].columns)
    for frame in boa_frames[1:]:
        if set(frame.columns) != columns:
            raise ValueError(
                f"seeds score different experiments: "
                f"{sorted(columns)} vs {sorted(frame.columns)}"
            )

    stacked = pd.concat(boa_frames, axis=0)
    return stacked.groupby(level=0).mean()


def compare_per_scene_boa(
    boa_a: pd.DataFrame,
    boa_b: pd.DataFrame,
) -> tuple[PairedComparison, ...]:
    """Wilcoxon signed-rank test per experiment, paired by scene identifier.

    Only scenes with a defined value for both models enter the test for a
    given experiment; a scene missing or undefined in either frame is
    dropped for that experiment, the same restriction a single model's own
    pairable scene count already applies.
    """
    if set(boa_a.columns) != set(boa_b.columns):
        raise ValueError(
            f"the two frames score different experiments: "
            f"{sorted(boa_a.columns)} vs {sorted(boa_b.columns)}"
        )

    results = []
    for experiment in boa_a.columns:
        joined = pd.DataFrame({"a": boa_a[experiment], "b": boa_b[experiment]}).dropna()

        if len(joined) == 0:
            results.append(
                PairedComparison(
                    experiment=experiment,
                    n_pairs=0,
                    statistic=float("nan"),
                    p_value=float("nan"),
                    median_difference=float("nan"),
                )
            )
            continue

        difference = joined["a"] - joined["b"]
        if (difference == 0).all():
            statistic, p_value = 0.0, 1.0
        else:
            statistic, p_value = wilcoxon(joined["a"], joined["b"])

        results.append(
            PairedComparison(
                experiment=experiment,
                n_pairs=len(joined),
                statistic=float(statistic),
                p_value=float(p_value),
                median_difference=float(difference.median()),
            )
        )

    return tuple(results)


def compare_across_seeds(
    boa_frames_a: tuple,
    boa_frames_b: tuple,
) -> tuple[PairedComparison, ...]:
    """Average each architecture's per-scene metric across its seeds, then
    run the paired Wilcoxon test on the averaged values.

    This is the note 2.10 comparison as actually run across the full set of
    seeds: seeds are averaged into one representative value per scene
    before pairing, so the test pairs by scene, not by seed.
    """
    averaged_a = average_across_seeds(boa_frames_a)
    averaged_b = average_across_seeds(boa_frames_b)
    return compare_per_scene_boa(averaged_a, averaged_b)