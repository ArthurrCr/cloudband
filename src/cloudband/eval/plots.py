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
PHASE2_MODELS = ("regnety", "convnext", "ocm", "swin", "ocm published")
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


def plot_seed_boa(seed_table: pd.DataFrame, title: str) -> Figure:
    """Whole-split BOA per model: each seed as a small dot, the seed mean labelled.

    seed_table is evaluation.seed_table's output, indexed by (model, experiment), with
    a "mean" column and one "seed N" column per seed. A model without seeds (the
    published one) has the same value in every seed column and is drawn as one diamond.
    """
    models = [m for m in PHASE2_MODELS if m in seed_table.index.get_level_values(0)]
    models += sorted(set(seed_table.index.get_level_values(0)) - set(models))
    seed_columns = [c for c in seed_table.columns if c.startswith("seed ")]
    rows = list(range(len(models)))[::-1]

    fig, axes = plt.subplots(1, 3, figsize=(13, 0.55 * len(models) + 1.6), sharey=True)
    for ax, experiment in zip(axes, EXPERIMENT_ORDER):
        lowest, highest = 100.0, 0.0
        for y, model in zip(rows, models):
            values = seed_table.loc[(model, experiment), seed_columns].to_numpy(float) * 100
            mean = float(seed_table.loc[(model, experiment), "mean"]) * 100
            color = _model_color(model)
            lowest, highest = min(lowest, values.min()), max(highest, values.max())
            if "published" in model:
                ax.scatter([mean], [y], marker="D", s=70, facecolor="white",
                           edgecolor=color, linewidth=2, zorder=3)
            else:
                ax.scatter(values, [y] * len(values), s=22, color=color, alpha=0.45,
                           linewidth=0, zorder=2)
                ax.scatter([mean], [y], s=90, color=color, edgecolor="white",
                           linewidth=2, zorder=3)
            ax.annotate(f"{mean:.2f}", (mean, y), xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=8.5, color=INK)
        pad = max(0.15, 0.08 * (highest - lowest))
        ax.set_xlim(lowest - pad, highest + pad)
        ax.set_ylim(-0.7, len(models) - 0.3)
        ax.set_title(experiment, color=INK, fontsize=11)
        ax.set_xlabel("BOA (%)", color=MUTED)
        _quiet(ax)
    axes[0].set_yticks(rows, models, color=INK)
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.text(0.01, 0.005, "small dots: seeds  ·  large dot: seed mean  ·  "
             "hollow diamond: published OmniCloudMask (no seeds)",
             color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    plt.close(fig)
    return fig


def _comparison_label(title: str) -> str:
    name = title.split(",")[0].replace(" alone", "")
    return name.replace(" vs ", " − ")


def plot_paired_differences(
    tables_by_level: Mapping[str, Mapping[str, pd.DataFrame]],
    comparisons: Sequence[str],
    title: str,
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
        ax.set_title(experiment, color=INK, fontsize=11)
        ax.set_xlabel("difference in BOA (points)", color=MUTED)
        ax.set_ylim(-0.6, count - 0.4)
        _quiet(ax)
    axes[0].set_yticks(
        list(range(count))[::-1], [_comparison_label(c) for c in comparisons], color=INK
    )
    if len(tables_by_level) > 1:
        axes[-1].legend(frameon=False, loc="lower right", labelcolor=INK, fontsize=9)
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.text(0.01, 0.005, "dot: mean difference per ROI, averaged over seeds  ·  "
             "line: 95% interval resampling ROIs  ·  right of zero favours the first model",
             color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
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


def plot_mean_vs_published(
    seed_tables: Mapping[str, pd.DataFrame],
    models: Sequence[str],
    published: str,
    title: str,
) -> Figure:
    """Mean BOA of the seeds ± one SD per trained model, the published model as a line.

    seed_tables maps a level name (L1C, L2A) to evaluation.seed_table's output; each
    level is one row of panels, each experiment one column. The value next to each
    dot is its distance to the published model, in BOA points.
    """
    levels = list(seed_tables)
    fig, axes = plt.subplots(
        len(levels), 3, figsize=(13, (0.55 * len(models) + 1.3) * len(levels) + 0.6),
        sharey=True, squeeze=False,
    )
    rows = list(range(len(models)))[::-1]
    for row_axes, level in zip(axes, levels):
        table = seed_tables[level]
        for ax, experiment in zip(row_axes, EXPERIMENT_ORDER):
            reference = float(table.loc[(published, experiment), "mean"]) * 100
            lowest = highest = reference
            for y, model in zip(rows, models):
                mean = float(table.loc[(model, experiment), "mean"]) * 100
                sd = float(table.loc[(model, experiment), "sd"]) * 100
                sd = 0.0 if np.isnan(sd) else sd
                color = _model_color(model)
                ax.plot([mean - sd, mean + sd], [y, y], color=color, linewidth=2.5,
                        solid_capstyle="round", zorder=2)
                ax.scatter([mean], [y], s=80, color=color, edgecolor="white", linewidth=2,
                           zorder=3)
                ax.annotate(f"{mean - reference:+.2f}", (mean, y), xytext=(0, 9),
                            textcoords="offset points", ha="center", fontsize=8.5, color=INK)
                lowest, highest = min(lowest, mean - sd), max(highest, mean + sd)
            ax.axvline(reference, color=REFERENCE, linewidth=1.5, linestyle=(0, (4, 3)),
                       zorder=1)
            ax.annotate(f"published {reference:.2f}", (reference, len(models) - 0.45),
                        xytext=(4, 0), textcoords="offset points", fontsize=8.5,
                        color=MUTED, va="center")
            pad = max(0.15, 0.12 * (highest - lowest))
            ax.set_xlim(lowest - pad, highest + pad * 2.2)
            ax.set_ylim(-0.6, len(models) - 0.2)
            ax.set_title(f"{experiment}, {level}", color=INK, fontsize=11)
            _quiet(ax)
        row_axes[0].set_yticks(rows, list(models), color=INK)
    for ax in axes[-1]:
        ax.set_xlabel("BOA (%)", color=MUTED)
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.text(0.01, 0.005, "dot: mean of the seeds  ·  line: ± one SD across seeds  ·  "
             "dashed: published model  ·  number: difference to it, in BOA points",
             color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
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
