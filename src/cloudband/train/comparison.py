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