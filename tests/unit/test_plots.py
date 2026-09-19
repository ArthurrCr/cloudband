import pandas as pd
import pytest

from cloudband.eval.palette import PASTEL_PALETTE, assign_colors
from cloudband.eval.plots import plot_experiment_comparison
from cloudband.eval.report import compare


def test_assign_colors_gives_each_model_a_distinct_colour():
    colors = assign_colors(["v1", "v2", "v3"])
    assert len(colors) == 3
    assert len(set(colors.values())) == 3


def test_assign_colors_is_stable_across_calls():
    first = assign_colors(["v1", "v2"])
    second = assign_colors(["v1", "v2"])
    assert first == second


def test_assign_colors_raises_when_out_of_palette_colours():
    too_many = [f"v{i}" for i in range(len(PASTEL_PALETTE) + 1)]
    with pytest.raises(ValueError):
        assign_colors(too_many)


def fake_scores(clear, cloud, shadow):
    return pd.DataFrame(
        {"boa": [clear, cloud, shadow]},
        index=pd.Index(["clear", "cloud", "shadow"], name="experiment"),
    )


def test_plot_experiment_comparison_makes_one_figure_per_experiment():
    frames = {
        "ocm-model1.0": fake_scores(94.0, 92.0, 88.0),
        "ocm-model4.0": fake_scores(95.0, 93.0, 90.0),
    }
    comparison = compare(frames, column="boa")
    colors = assign_colors(list(comparison.columns))

    figures = plot_experiment_comparison(comparison, colors)

    assert set(figures) == {"clear", "cloud", "shadow"}
    for figure in figures.values():
        assert len(figure.axes) == 1


def test_plot_experiment_comparison_draws_one_bar_per_model_in_its_colour():
    frames = {
        "ocm-model1.0": fake_scores(94.0, 92.0, 88.0),
        "ocm-model4.0": fake_scores(95.0, 93.0, 90.0),
    }
    comparison = compare(frames, column="boa")
    colors = assign_colors(list(comparison.columns))

    figures = plot_experiment_comparison(comparison, colors)

    ax = figures["clear"].axes[0]
    bars = ax.patches
    assert len(bars) == 2
    heights = sorted(bar.get_height() for bar in bars)
    assert heights == [94.0, 95.0]

    drawn_colors = {bar.get_facecolor() for bar in bars}
    from matplotlib.colors import to_rgba

    expected_colors = {to_rgba(colors[model]) for model in comparison.columns}
    assert drawn_colors == expected_colors


def test_plot_experiment_comparison_rejects_a_model_missing_a_colour():
    frames = {
        "ocm-model1.0": fake_scores(94.0, 92.0, 88.0),
        "ocm-model4.0": fake_scores(95.0, 93.0, 90.0),
    }
    comparison = compare(frames, column="boa")
    incomplete_colors = {"ocm-model1.0": "#A8D8B9"}

    with pytest.raises(ValueError):
        plot_experiment_comparison(comparison, incomplete_colors)