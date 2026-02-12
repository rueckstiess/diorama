"""Diorama: 2D/3D embedding visualization for JSON documents."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from diorama.perspectives import ColoringPerspective

if TYPE_CHECKING:
    import plotly.graph_objects as go

__all__ = ["show", "reduce", "app", "ColoringPerspective"]


def show(
    embeddings: np.ndarray,
    documents: list[dict],
    *,
    color_by: str | list[str] | None = None,
    filter: dict | None = None,
    n_components: int = 2,
    method: str = "umap",
    color_type_overrides: dict[str, str] | None = None,
    categorical_threshold: int = 20,
    max_categorical_traces: int = 20,
    color_scheme: str = "light",
    height: int | None = None,
    show_progress: bool = False,
    subsample: int | None = None,
    umap_kwargs: dict | None = None,
    output_path: str | None = None,
) -> go.Figure:
    """Visualize embeddings colored by document fields.

    Args:
        embeddings: (N, D) high-dimensional or (N, 2)/(N, 3) pre-reduced embeddings.
        documents: List of N JSON documents (Python dicts).
        color_by: Field path(s) in dot notation. If None, auto-discovers top fields
            by coverage. If a string, colors by that single field. If a list, creates
            a dropdown with those fields.
        filter: MongoDB-style query dict to filter documents. Supported operators:
            $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $exists, $regex,
            $and, $or, $not, $nor.
        n_components: 2 or 3. Ignored if embeddings are already 2D or 3D.
        method: Dimensionality reduction method ("umap" or "tsne").
        color_type_overrides: Dict mapping field path to "categorical" or "continuous".
        categorical_threshold: Numeric fields with fewer unique values than this
            are treated as categorical.
        max_categorical_traces: Max categories before grouping as "Other".
        color_scheme: "light" or "dark".
        height: Figure height in pixels. None uses Plotly's default.
        show_progress: Show progress bar during dimensionality reduction.
        subsample: Fit reducer on this many random points, then transform all.
            Much faster for large datasets. Only supported for UMAP.
        umap_kwargs: Extra kwargs passed to the UMAP/T-SNE constructor.
        output_path: If given, save the figure as an HTML file.

    Returns:
        Plotly Figure object.
    """
    from diorama.fields import build_perspectives, top_field_paths
    from diorama.hover import create_hover_text
    from diorama.reduction import reduce_embeddings
    from diorama.viz import create_figure

    if len(embeddings) != len(documents):
        raise ValueError(
            f"embeddings length ({len(embeddings)}) must match documents length ({len(documents)})"
        )

    # 1. Reduce all embeddings first (stable positions)
    if embeddings.shape[1] > 3:
        embedding_low = reduce_embeddings(
            embeddings,
            n_components=n_components,
            method=method,
            show_progress=show_progress,
            subsample=subsample,
            **(umap_kwargs or {}),
        )
    elif embeddings.shape[1] in (2, 3):
        embedding_low = embeddings
    else:
        raise ValueError(f"Embeddings must have at least 2 dimensions, got {embeddings.shape[1]}")

    # 2. Apply filter (on pre-reduced points for stable positions)
    if filter is not None:
        from diorama.query import filter_documents

        documents, mask = filter_documents(documents, filter)
        embedding_low = embedding_low[mask]

    if len(documents) == 0:
        import plotly.graph_objects as go

        return go.Figure()

    # 3. Determine color_by fields
    if color_by is None:
        color_by = top_field_paths(documents)
    elif isinstance(color_by, str):
        color_by = [color_by]

    if not color_by:
        import plotly.graph_objects as go

        return go.Figure()

    # 4. Build perspectives
    perspectives = build_perspectives(
        documents,
        color_by,
        color_type_overrides=color_type_overrides,
        categorical_threshold=categorical_threshold,
    )

    # 5. Generate hover text
    hover_text = create_hover_text(documents)

    # 6. Create figure
    method_label = method.upper() if method in ("umap", "tsne") else method
    if method_label == "TSNE":
        method_label = "T-SNE"

    fig = create_figure(
        embedding_low,
        perspectives,
        hover_text,
        color_scheme=color_scheme,
        max_categorical_traces=max_categorical_traces,
        method_label=method_label,
        height=height,
    )

    if output_path:
        fig.write_html(output_path)

    return fig


def reduce(
    embeddings: np.ndarray,
    *,
    n_components: int = 2,
    method: str = "umap",
    show_progress: bool = False,
    subsample: int | None = None,
    **kwargs,
) -> np.ndarray:
    """Reduce high-dimensional embeddings to 2D or 3D.

    Convenience function for caching the (expensive) reduction step.
    Pass the result to show() for fast iteration with different filters/colors.

    Args:
        embeddings: (N, D) array where D > 3.
        n_components: 2 or 3.
        method: "umap" or "tsne".
        show_progress: Show progress bar during reduction.
        subsample: Fit reducer on this many random points, then transform all.
            Much faster for large datasets. Only supported for UMAP.
        **kwargs: Passed to the underlying reducer constructor.

    Returns:
        (N, n_components) numpy array.
    """
    from diorama.reduction import reduce_embeddings

    return reduce_embeddings(
        embeddings,
        n_components=n_components,
        method=method,
        show_progress=show_progress,
        subsample=subsample,
        **kwargs,
    )


def app(
    embeddings: np.ndarray,
    documents: list[dict],
    *,
    method: str = "umap",
    n_components: int = 2,
    host: str = "127.0.0.1",
    port: int = 8050,
    debug: bool = False,
    subsample: int | None = None,
    umap_kwargs: dict | None = None,
) -> None:
    """Launch interactive Dash app for exploring embeddings.

    Requires the 'dash' optional dependency: pip install diorama[dash]

    Args:
        embeddings: (N, D) or pre-reduced (N, 2)/(N, 3).
        documents: List of N JSON documents.
        method: Reduction method.
        n_components: Initial dimensionality (user can toggle in the app).
        host: Server host.
        port: Server port.
        debug: Dash debug mode.
        subsample: Fit reducer on this many random points, then transform all.
        umap_kwargs: Extra kwargs passed to the reducer.
    """
    from diorama.dashboard import create_app

    dash_app = create_app(
        embeddings,
        documents,
        method=method,
        n_components=n_components,
        subsample=subsample,
        umap_kwargs=umap_kwargs,
    )
    dash_app.run(host=host, port=port, debug=debug)
