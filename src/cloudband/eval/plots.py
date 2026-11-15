"""Charts comparing models on each experiment.

The phase 0 bars use the shared pastel palette, one colour per model version.
The phase 2 figures show the seeds and the paired differences behind the tables
of eval.phase2, one panel per experiment (clear, cloud, shadow).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator


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
        plt.close(fig)
        figures[experiment] = fig

    return figures

# Phase 2 figures. Blue is the OCM family, orange is Swin, gray is the published
# OmniCloudMask, a reference outside the controlled comparison. Text stays in ink.
INK = "#0b0b0b"
MUTED = "#52514e"
REFERENCE = "#8a8984"
GRID = "#e4e3df"
OCM_COLOR = "#2a78d6"
SWIN_COLOR = "#eb6834"
EXPERIMENT_ORDER = ("clear", "cloud", "shadow")


def _model_color(model: str) -> str:
    if model == "swin":
        return SWIN_COLOR
    if "published" in model:
        return REFERENCE
    return OCM_COLOR


def _quiet(ax, axis: str = "x") -> None:
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if axis == "x":
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))


def plot_all_models(
    seed_table: pd.DataFrame,
    groups: Sequence[Sequence[str]],
    labels: Mapping[str, str],
    title: str,
    metric_label: str = "Median BOA per patch (%)",
) -> Figure:
    """BOA of every model on one axis per experiment.

    seed_table is evaluation.seed_table's output, indexed by (model, experiment).
    groups lists the models top to bottom in blocks (the published versions, then the
    trained models), separated by a rule. A model with seeds shows each seed as a
    small dot and the mean as a large one; a model without seeds (a published
    version) is one diamond. Every point is named on the axis, and its value is
    written next to it.
    """
    order = [model for group in groups for model in group]
    rows, y, rules = {}, 0.0, []
    for number, group in enumerate(groups):
        if number:
            rules.append(y + 0.25)
            y += 0.5
        for model in group:
            rows[model] = y
            y += 1
    top = y - 1
    seed_columns = [c for c in seed_table.columns if c.startswith("seed ")]

    fig, axes = plt.subplots(1, 3, figsize=(14, 0.48 * y + 1.6), sharey=True)
    for ax, experiment in zip(axes, EXPERIMENT_ORDER):
        means = [float(seed_table.loc[(m, experiment), "mean"]) * 100 for m in order]
        spread = [seed_table.loc[(m, experiment), seed_columns].to_numpy(float) * 100 for m in order]
        lowest = min(min(v.min() for v in spread), min(means))
        highest = max(max(v.max() for v in spread), max(means))
        for model, mean, values in zip(order, means, spread):
            row = top - rows[model]
            color = _model_color(model)
            if np.ptp(values) == 0:
                ax.scatter([mean], [row], marker="D", s=60, color=REFERENCE, edgecolor="white",
                           linewidth=1.5, zorder=3)
            else:
                ax.scatter(values, [row] * len(values), s=18, color=color, alpha=0.45,
                           linewidth=0, zorder=2)
                ax.scatter([mean], [row], s=80, color=color, edgecolor="white", linewidth=2,
                           zorder=3)
            ax.annotate(f"{mean:.2f}", (max(mean, values.max()), row), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=8.5, color=INK)
        for rule in rules:
            ax.axhline(top - rule, color=GRID, linewidth=1.2)
        pad = max(0.2, 0.1 * (highest - lowest))
        ax.set_xlim(lowest - pad, highest + pad * 3)
        ax.set_ylim(-0.6, top + 0.6)
        ax.set_title(experiment, color=INK, fontsize=11)
        ax.set_xlabel(metric_label, color=MUTED)
        _quiet(ax)
    axes[0].set_yticks([top - rows[m] for m in order], [labels.get(m, m) for m in order],
                       color=INK)
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.text(0.01, 0.005, "gray diamond: published model (no seeds)  ·  small dots: the seeds of "
             "a trained model  ·  large dot and number: their mean",
             color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    plt.close(fig)
    return fig


# Bars of the share figure: the published versions as a gray ramp (1.0 light to
# 4.0 dark), the trained OCM as a blue ramp (each backbone light, the ensemble
# dark), Swin orange.
PUBLISHED_RAMP = ("#d3d2cc", "#b0afa8", "#8a8984", "#5f5e59")
OCM_RAMP = {"regnety": "#86b6ef", "convnext": "#5598e7", "ocm": "#1c5cab"}


def share_colors(models: Sequence[str]) -> dict[str, str]:
    """Bar colour of each model of the share figure, by family."""
    published = [m for m in models if "published" in m]
    colors = {m: PUBLISHED_RAMP[min(i, len(PUBLISHED_RAMP) - 1)] for i, m in enumerate(published)}
    for model in models:
        if model == "swin":
            colors[model] = SWIN_COLOR
        elif model in OCM_RAMP:
            colors[model] = OCM_RAMP[model]
    return colors


def plot_boa_by_share(
    table: pd.DataFrame,
    models: Sequence[str],
    labels: Mapping[str, str],
    title: str,
) -> Figure:
    """BOA of every model in each range of class share, for one experiment.

    table is one experiment of evaluation.boa_by_share, indexed by (bin, model), with
    the patch count n, the mean over seeds of the median BOA per patch and its SD. One group of bars per
    range, one bar per model in the given order, the number of patches under each
    group, and the BOA written above each bar. The axis runs from 0 to 100%; a model
    with seeds carries a line of one SD.
    """
    bins = list(dict.fromkeys(table.index.get_level_values(0)))
    colors = share_colors(models)
    width = 0.84 / len(models)

    fig, ax = plt.subplots(figsize=(14, 6))
    for index, model in enumerate(models):
        positions, heights, spreads = [], [], []
        for group, bin_label in enumerate(bins):
            if (bin_label, model) not in table.index:
                continue
            row = table.loc[(bin_label, model)]
            if np.isnan(row["mean"]):
                continue
            positions.append(group - 0.42 + width * (index + 0.5))
            heights.append(row["mean"] * 100)
            spreads.append(0.0 if np.isnan(row["sd"]) else row["sd"] * 100)
        ax.bar(positions, heights, width=width, color=colors[model],
               edgecolor="white", linewidth=1, label=labels.get(model, model), zorder=2)
        if any(spreads):
            ax.errorbar(positions, heights, yerr=spreads, fmt="none", ecolor=INK,
                        elinewidth=1, capsize=0, zorder=3)
        for position, height, spread in zip(positions, heights, spreads):
            ax.annotate(f"{height:.2f}", (position, height + spread), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", rotation=90,
                        fontsize=7.5, color=INK)
    sizes = table.groupby(level=0, sort=False)["n"].first()
    ax.set_xticks(range(len(bins)),
                  [f"{b}\nn = {int(sizes[b])} patches" for b in bins], color=INK)
    ax.set_xlim(-0.55, len(bins) - 0.45)
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Median BOA per patch (%)", color=MUTED)
    ax.set_xlabel("share of the patch's annotated pixels in the class", color=MUTED)
    _quiet(ax, axis="y")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0), labelcolor=INK,
              fontsize=9)
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.text(0.01, 0.005, "trained models: mean of the 5 seeds, line = ± one SD  ·  "
             "median over the patches of the range that hold both the class and its "
             "complement (CloudSEN12 convention); n counts those patches",
             color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    plt.close(fig)
    return fig

