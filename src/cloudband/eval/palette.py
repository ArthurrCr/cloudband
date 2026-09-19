"""Pastel colour palette shared across the project's comparison charts.

Colours are assigned per model, not per class: every chart in this project
compares the same set of models against each other on the same experiments,
so a model keeps the same colour across every experiment's chart. Keeping
this palette in one place, separate from any plotting code, means any part
of the project can look up a model's colour without importing matplotlib.
"""

from __future__ import annotations

from collections.abc import Sequence

PASTEL_PALETTE: tuple[str, ...] = (
    "#A8D8B9",  # pastel green
    "#F7DC8F",  # pastel yellow
    "#AEE6F5",  # pastel blue
    "#F5B7B1",  # pastel red
    "#D7BDE2",  # pastel purple
    "#F8C9A0",  # pastel orange
    "#A9CCE3",  # pastel steel blue
    "#C9E4B5",  # pastel lime
)


def assign_colors(
    model_ids: Sequence[str], palette: tuple[str, ...] = PASTEL_PALETTE
) -> dict[str, str]:
    """Assign one palette colour to each model id, in the order given.

    Raises rather than reusing a colour for two different models once the
    palette runs out, so a chart never silently gives two models the same
    colour.
    """
    if len(model_ids) > len(palette):
        raise ValueError(
            f"{len(model_ids)} models but only {len(palette)} palette colours; "
            "extend PASTEL_PALETTE"
        )
    return dict(zip(model_ids, palette))