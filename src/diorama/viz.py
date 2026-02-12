"""Plotly figure construction for embedding visualizations.

Adapted from the reference implementation in index-advisor-usyd. Creates
interactive 2D/3D scatter plots with dropdown menus to switch between
coloring perspectives.
"""

from collections import Counter

import numpy as np
import plotly.colors
import plotly.graph_objects as go

from diorama.perspectives import ColoringPerspective


def create_figure(
    embedding: np.ndarray,
    perspectives: list[ColoringPerspective],
    hover_text: list[str],
    *,
    color_scheme: str = "light",
    max_categorical_traces: int = 20,
    method_label: str = "UMAP",
    height: int | None = None,
) -> go.Figure:
    """Create interactive Plotly figure with dropdown for switching coloring perspectives.

    Categorical perspectives create one trace per unique value (toggled as a group).
    Continuous perspectives create a single trace with a colorscale.

    Args:
        embedding: (N, 2) or (N, 3) reduced embeddings.
        perspectives: Coloring perspectives for the dropdown menu.
        hover_text: HTML hover text for each point, length N.
        color_scheme: "light" or "dark" theme.
        max_categorical_traces: Max categories before grouping rest as "Other".
        method_label: Label for axis titles (e.g. "UMAP", "T-SNE").

    Returns:
        Plotly Figure object.
    """
    if embedding.ndim != 2 or embedding.shape[1] not in (2, 3):
        raise ValueError(f"embedding must have shape (N, 2) or (N, 3), got {embedding.shape}")

    n_dims = embedding.shape[1]
    x = embedding[:, 0]
    y = embedding[:, 1]
    z = embedding[:, 2] if n_dims == 3 else None

    # Theme
    if color_scheme == "dark":
        template = "plotly_dark"
        dropdown_bgcolor = "#2a2a2a"
        dropdown_font_color = "#e0e0e0"
        dropdown_border_color = "#555"
    else:
        template = "plotly_white"
        dropdown_bgcolor = "#ffffff"
        dropdown_font_color = "#333333"
        dropdown_border_color = "#cccccc"

    fig = go.Figure()
    trace_counts = []

    for i, persp in enumerate(perspectives):
        visible = i == 0

        if persp.color_type == "categorical":
            n_traces = _add_categorical_traces(
                fig, x, y, z, persp, hover_text, visible, max_categorical_traces
            )
            trace_counts.append(n_traces)
        elif persp.color_type == "continuous":
            n_traces = _add_continuous_trace(fig, x, y, z, persp, hover_text, visible)
            trace_counts.append(n_traces)

    # Dropdown buttons
    buttons = []
    trace_offset = 0
    for i, persp in enumerate(perspectives):
        n_traces = trace_counts[i]
        if n_traces == 0:
            continue

        visible_list = [False] * len(fig.data)
        for j in range(trace_offset, trace_offset + n_traces):
            visible_list[j] = True

        buttons.append(
            dict(
                label=persp.name,
                method="update",
                args=[
                    {"visible": visible_list},
                    {"title": f"{n_dims}D Embedding (colored by {persp.name})"},
                ],
            )
        )
        trace_offset += n_traces

    layout_kwargs = {
        "template": template,
        "title": f"{n_dims}D Embedding (colored by {perspectives[0].name})" if perspectives else "",
        "font": dict(size=12),
        "title_font_size": 16,
        "autosize": True,
        **({"height": height} if height is not None else {}),
    }

    if buttons:
        layout_kwargs["updatemenus"] = [
            dict(
                buttons=buttons,
                direction="up",
                showactive=True,
                x=0.02,
                xanchor="left",
                y=0.02,
                yanchor="bottom",
                bgcolor=dropdown_bgcolor,
                font=dict(color=dropdown_font_color),
                bordercolor=dropdown_border_color,
                borderwidth=1,
            )
        ]

    if n_dims == 3:
        layout_kwargs["scene"] = dict(
            xaxis_title=f"{method_label} 1",
            yaxis_title=f"{method_label} 2",
            zaxis_title=f"{method_label} 3",
        )
    else:
        layout_kwargs["xaxis"] = dict(title=f"{method_label} 1")
        layout_kwargs["yaxis"] = dict(title=f"{method_label} 2")

    fig.update_layout(**layout_kwargs)
    return fig


def _add_categorical_traces(
    fig: go.Figure,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray | None,
    persp: ColoringPerspective,
    hover_text: list[str],
    visible: bool,
    max_traces: int,
) -> int:
    """Add categorical traces to figure. Returns number of traces added."""
    values = persp.values

    # Count and cap categories
    counts = Counter(values)
    if len(counts) > max_traces:
        top_values = {v for v, _ in counts.most_common(max_traces)}
        values = [v if v in top_values else "Other" for v in values]
        counts = Counter(values)

    # Sort: by count descending, "Other" last
    sorted_categories = sorted(
        counts.keys(),
        key=lambda v: (v == "Other", -counts[v], str(v)),
    )

    # Build color map if not provided
    color_map = persp.color_map
    if not color_map:
        palette = plotly.colors.qualitative.Alphabet
        color_map = {}
        for idx, cat in enumerate(sorted_categories):
            if cat == "Other":
                color_map[cat] = "#808080"
            else:
                color_map[cat] = palette[idx % len(palette)]

    hover_arr = np.array(hover_text, dtype=object)

    for cat in sorted_categories:
        mask = np.array([v == cat for v in values])
        color = color_map.get(str(cat), "#808080")

        scatter_kwargs = dict(
            mode="markers",
            name=str(cat),
            text=hover_arr[mask],
            hovertemplate="%{text}<extra></extra>",
            marker=dict(size=4, opacity=0.6, color=color),
            visible=visible,
        )

        if z is not None:
            fig.add_trace(go.Scatter3d(x=x[mask], y=y[mask], z=z[mask], **scatter_kwargs))
        else:
            fig.add_trace(go.Scatter(x=x[mask], y=y[mask], **scatter_kwargs))

    return len(sorted_categories)


def _add_continuous_trace(
    fig: go.Figure,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray | None,
    persp: ColoringPerspective,
    hover_text: list[str],
    visible: bool,
) -> int:
    """Add a continuous-colored trace to figure. Returns number of traces added (0 or 1)."""
    valid_mask = np.array([v is not None for v in persp.values])
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return 0

    valid_values = np.array([persp.values[i] for i in valid_indices], dtype=float)
    hover_arr = np.array(hover_text, dtype=object)

    cmin = persp.cmin if persp.cmin is not None else float(np.nanmin(valid_values))
    cmax = persp.cmax if persp.cmax is not None else float(np.nanmax(valid_values))
    colorbar_title = persp.colorbar_title or persp.name

    scatter_kwargs = dict(
        mode="markers",
        name=persp.name,
        text=hover_arr[valid_indices],
        hovertemplate="%{text}<extra></extra>",
        marker=dict(
            size=4,
            opacity=0.7,
            color=valid_values,
            colorscale=persp.colorscale,
            cmin=cmin,
            cmax=cmax,
            colorbar=dict(title=colorbar_title, x=1.02),
            showscale=True,
        ),
        visible=visible,
    )

    if z is not None:
        fig.add_trace(
            go.Scatter3d(
                x=x[valid_indices], y=y[valid_indices], z=z[valid_indices], **scatter_kwargs
            )
        )
    else:
        fig.add_trace(go.Scatter(x=x[valid_indices], y=y[valid_indices], **scatter_kwargs))

    return 1
