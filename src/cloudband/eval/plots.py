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


def _comparison_label(title: str) -> str:
    name = title.split(",")[0].replace(" alone", "")
    return name.replace(" vs ", " − ")


def plot_paired_differences(
    tables_by_level: Mapping[str, Mapping[str, pd.DataFrame]],
    comparisons: Sequence[str],
    title: str,
    labels: Mapping[str, str] | None = None,
) -> Figure:
    """Mean paired difference and its 95% interval, in BOA points, per comparison.

    tables_by_level maps a level name ("L1C", "L2A") to evaluation.report's output.
    Each comparison is a key of those reports; a difference above zero favours the
    model named first. The first level is drawn filled, the second hollow.
    """
    styles = (("o", INK, INK), ("o", MUTED, "white"), ("s", MUTED, MUTED))
    if len(tables_by_level) > len(styles):
        raise ValueError(f"at most {len(styles)} levels")
    count = len(comparisons)
    offsets = [0.0] if len(tables_by_level) == 1 else [
        0.16 - 0.32 * i / (len(tables_by_level) - 1) for i in range(len(tables_by_level))
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 0.62 * count + 1.8), sharey=True)
    for ax, experiment in zip(axes, EXPERIMENT_ORDER):
        ax.axvline(0, color=MUTED, linewidth=1, zorder=1)
        for (level, tables), offset, (marker, edge, face) in zip(
            tables_by_level.items(), offsets, styles
        ):
            for row, name in enumerate(comparisons):
                y = count - 1 - row + offset
                line = tables[name].loc[experiment]
                mean = line["mean diff"] * 100
                low, high = line["ci95 low"] * 100, line["ci95 high"] * 100
                ax.plot([low, high], [y, y], color=edge, linewidth=2,
                        solid_capstyle="round", zorder=2)
                ax.scatter([mean], [y], marker=marker, s=60, color=face, edgecolor=edge,
                           linewidth=2, zorder=3, label=level if row == 0 else None)
                ax.annotate(f"{mean:+.2f}", (high, y), xytext=(6, 0), textcoords="offset points",
                            va="center", fontsize=8, color=edge)
        low, high = ax.get_xlim()
        ax.set_xlim(low, high + 0.15 * (high - low))
        ax.set_title(experiment, color=INK, fontsize=11)
        ax.set_xlabel("difference in BOA (points)", color=MUTED)
        ax.set_ylim(-0.6, count - 0.4)
        _quiet(ax)
    axes[0].set_yticks(
        list(range(count))[::-1],
        [(labels or {}).get(c, _comparison_label(c)) for c in comparisons],
        color=INK,
    )
    if len(tables_by_level) > 1:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, loc="upper right",
                   ncol=len(tables_by_level), labelcolor=INK, fontsize=9)
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.text(0.01, 0.005, "dot: mean difference per ROI, averaged over seeds  ·  "
             "line: 95% interval resampling ROIs  ·  right of zero favours the first model",
             color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    plt.close(fig)
    return fig


def plot_unit_differences(
    differences: pd.DataFrame, name_a: str, name_b: str, title: str, unit: str = "ROIs"
) -> Figure:
    """Histogram of the per-unit difference A − B, one panel per experiment.

    Bars right of zero (A better) are blue, left of zero (B better) orange, and the
    bar holding zero gray, with the share of units where A is better written above.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, experiment in zip(axes, EXPERIMENT_ORDER):
        values = differences[experiment].dropna().to_numpy() * 100
        reach = max(float(np.percentile(np.abs(values), 99)), 0.5)
        width = reach / 12
        edges = np.arange(-12.5, 13.5) * width
        heights, _ = np.histogram(np.clip(values, edges[0], edges[-1]), bins=edges)
        centers = (edges[:-1] + edges[1:]) / 2
        colors = [
            REFERENCE if abs(c) < width / 2 else (OCM_COLOR if c > 0 else SWIN_COLOR)
            for c in centers
        ]
        ax.bar(centers, heights, width=width * 0.86, color=colors, linewidth=0)
        ax.axvline(0, color=MUTED, linewidth=1)
        better = float((values > 0).mean()) * 100
        ax.set_title(experiment, color=INK, fontsize=11)
        ax.text(0.98, 0.95, f"{name_a} better in {better:.0f}% of {len(values)} {unit}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 2}, zorder=4)
        ax.set_xlabel(f"{name_a} − {name_b}, BOA points", color=MUTED)
        _quiet(ax, axis="y")
    axes[0].set_ylabel(unit, color=MUTED)
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.text(0.01, 0.005, f"blue: {name_a} better  ·  orange: {name_b} better  ·  gray: the bar around zero  ·  "
             "values past the edges are counted in the outermost bars",
             color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    plt.close(fig)
    return fig


def plot_unit_scatter(
    trained: Mapping[str, pd.DataFrame],
    reference: pd.DataFrame,
    reference_label: str,
    title: str,
    unit: str = "ROIs",
) -> Figure:
    """BOA per unit of each trained model (y) against the reference model (x).

    One row of panels per trained model, one column per experiment. A point above
    the diagonal is a unit where the trained model does better.
    """
    names = list(trained)
    fig, axes = plt.subplots(len(names), 3, figsize=(13, 4.3 * len(names)), squeeze=False)
    for row_axes, name in zip(axes, names):
        color = _model_color(name.lower())
        for ax, experiment in zip(row_axes, EXPERIMENT_ORDER):
            joined = pd.DataFrame(
                {"x": reference[experiment], "y": trained[name][experiment]}
            ).dropna() * 100
            low = float(min(joined["x"].quantile(0.01), joined["y"].quantile(0.01)))
            low = max(0.0, low - 2)
            ax.plot([low, 100], [low, 100], color=MUTED, linewidth=1, zorder=1)
            ax.scatter(joined["x"].clip(lower=low), joined["y"].clip(lower=low), s=16,
                       color=color, alpha=0.55, linewidth=0, zorder=2)
            above = float((joined["y"] > joined["x"]).mean()) * 100
            ax.text(0.03, 0.97, f"{name} better in {above:.0f}% of {len(joined)} {unit}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=9, color=INK)
            ax.set_xlim(low, 100.5)
            ax.set_ylim(low, 100.5)
            ax.set_aspect("equal")
            ax.set_title(experiment, color=INK, fontsize=11)
            ax.set_xlabel(f"{reference_label} BOA (%)", color=MUTED)
            ax.set_ylabel(f"{name} BOA (%)", color=MUTED)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            for spine in ("left", "bottom"):
                ax.spines[spine].set_color(MUTED)
            ax.tick_params(colors=MUTED, length=0)
            ax.grid(color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.text(0.01, 0.005, "one point per ROI  ·  above the diagonal: the trained model does "
             "better  ·  points below the axis range are drawn at its edge",
             color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    plt.close(fig)
    return fig


def plot_all_models(
    seed_table: pd.DataFrame,
    groups: Sequence[Sequence[str]],
    labels: Mapping[str, str],
    title: str,
) -> Figure:
    """Whole-split BOA of every model on one axis per experiment.

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
        ax.set_xlabel("BOA (%)", color=MUTED)
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


def plot_versus_references(
    tables: Mapping[str, pd.DataFrame],
    models: Mapping[str, str],
    references: Sequence[str],
    labels: Mapping[str, str],
    title: str,
) -> Figure:
    """Each trained model minus each reference model, per unit, with its 95% interval.

    tables is one level's evaluation.report output, holding "{model} vs {reference},
    per ROI" for every key of models and every reference. One row per reference; the
    trained models are drawn side by side in each row, in their own colours, with the
    difference written next to the dot.
    """
    names = list(models)
    offsets = [0.0] if len(names) == 1 else [
        0.18 - 0.36 * i / (len(names) - 1) for i in range(len(names))
    ]
    rows = list(range(len(references)))[::-1]

    fig, axes = plt.subplots(1, 3, figsize=(14, 0.95 * len(references) + 1.9), sharey=True)
    for ax, experiment in zip(axes, EXPERIMENT_ORDER):
        ax.axvline(0, color=MUTED, linewidth=1, zorder=1)
        for name, offset in zip(names, offsets):
            color = _model_color(name.lower())
            for y, reference in zip(rows, references):
                line = tables[f"{name} vs {reference}, per ROI"].loc[experiment]
                mean = line["mean diff"] * 100
                low, high = line["ci95 low"] * 100, line["ci95 high"] * 100
                ax.plot([low, high], [y + offset] * 2, color=color, linewidth=2.5,
                        solid_capstyle="round", zorder=2)
                ax.scatter([mean], [y + offset], s=60, color=color, edgecolor="white",
                           linewidth=1.5, zorder=3, label=models[name] if y == rows[0] else None)
                ax.annotate(f"{mean:+.2f}", (high, y + offset), xytext=(6, 0),
                            textcoords="offset points", va="center", fontsize=8, color=INK)
        low, high = ax.get_xlim()
        ax.set_xlim(low, high + 0.18 * (high - low))
        ax.set_ylim(-0.6, len(references) - 0.4)
        ax.set_title(experiment, color=INK, fontsize=11)
        ax.set_xlabel("difference in BOA (points)", color=MUTED)
        _quiet(ax)
    axes[0].set_yticks(rows, [f"vs {labels.get(r, r)}" for r in references], color=INK)
    handles, legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend, frameon=False, loc="upper right", ncol=len(names),
               labelcolor=INK, fontsize=9)
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.text(0.01, 0.005, "dot: mean difference per ROI, averaged over seeds  ·  line: 95% "
             "interval resampling ROIs  ·  right of zero: the trained model does better",
             color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    plt.close(fig)
    return fig

