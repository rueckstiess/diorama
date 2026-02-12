"""Coloring perspective dataclass for embedding visualizations."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class ColoringPerspective:
    """One way of coloring points in a scatter plot.

    A perspective maps to one or more Plotly traces that are shown/hidden
    together via a dropdown menu.

    Attributes:
        name: Display name in the dropdown menu.
        color_type: "categorical" for discrete categories, "continuous" for numeric scale.
        values: Per-point coloring values (length N). Strings for categorical,
            floats for continuous.
        color_map: Mapping from categorical values to colors.
        colorscale: Plotly colorscale name for continuous perspectives (e.g. "Viridis", "Plasma").
        colorbar_title: Colorbar label for continuous perspectives.
        cmin: Min value for continuous color range (auto-computed if None).
        cmax: Max value for continuous color range (auto-computed if None).
    """

    name: str
    color_type: Literal["categorical", "continuous"]
    values: list
    color_map: dict[str, str] | None = None
    colorscale: str = "Viridis"
    colorbar_title: str | None = None
    cmin: float | None = None
    cmax: float | None = None
