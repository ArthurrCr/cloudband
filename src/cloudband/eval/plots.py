"""Bar charts comparing models on each experiment.

One chart per experiment (clear, cloud, shadow), one bar per model, coloured
by the project's shared pastel palette so a model keeps the same colour
across every chart.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure


def plot_experiment_comparison(
    comparison: pd.DataFrame,
    colors: dict[str, str],
    metric_label: str = "BOA (%)",
) -> dict[str, Figure]:
    """One bar chart per experiment, bars coloured per model.

    comparison is shaped like eval.report.compare's output: one row per
    experiment, one column per model. colors maps each column name to a
    palette colour; every column must have one, so a model is never plotted
    in an unassigned colour.
    """
    missing = sorted(set(comparison.columns) - set(colors))
    if missing:
        raise ValueError(f"no colour assigned for models {missing}")

    figures = {}
    for experiment in comparison.index:
        values = comparison.loc[experiment]
        bar_colors = [colors[model] for model in values.index]

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(values.index.astype(str), values.to_numpy(), color=bar_colors)
        ax.bar_label(bars, fmt="%.2f", padding=3)
        ax.set_title(experiment)
        ax.set_ylabel(metric_label)
        ax.set_ylim(0, 100)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        figures[experiment] = fig

    return figures