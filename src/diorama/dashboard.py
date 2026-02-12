"""Optional Dash app for interactive embedding exploration.

Requires the 'dash' extra: pip install diorama[dash]
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import dash


def create_app(
    embeddings: np.ndarray,
    documents: list[dict],
    *,
    method: str = "umap",
    n_components: int = 2,
    subsample: int | None = None,
    umap_kwargs: dict | None = None,
) -> dash.Dash:
    """Create and return a Dash app for interactive embedding exploration.

    Pre-computes both 2D and 3D reductions at startup for instant toggling.
    """
    try:
        import dash
        from dash import Input, Output, State, dcc, html
    except ImportError:
        raise ImportError(
            "Dash is required for the interactive app. "
            "Install it with: pip install diorama[dash]"
        ) from None

    import plotly.graph_objects as go

    from diorama.fields import build_perspectives, extract_field_paths
    from diorama.hover import create_hover_text
    from diorama.query import filter_documents
    from diorama.reduction import reduce_embeddings
    from diorama.viz import create_figure

    # Pre-compute reductions
    kwargs = umap_kwargs or {}
    if embeddings.shape[1] <= 3:
        # Already reduced — use as-is for matching dimension, skip the other
        embedding_cache = {embeddings.shape[1]: embeddings}
    else:
        embedding_cache = {
            2: reduce_embeddings(embeddings, n_components=2, method=method, subsample=subsample, **kwargs),
            3: reduce_embeddings(embeddings, n_components=3, method=method, subsample=subsample, **kwargs),
        }

    field_paths = extract_field_paths(documents)

    app = dash.Dash(__name__)

    app.layout = html.Div(
        [
            html.H1("Diorama", style={"marginBottom": "20px"}),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Color by:"),
                            dcc.Dropdown(
                                id="field-dropdown",
                                options=[{"label": p, "value": p} for p in field_paths],
                                value=field_paths[0] if field_paths else None,
                            ),
                        ],
                        style={"width": "300px", "display": "inline-block", "marginRight": "20px"},
                    ),
                    html.Div(
                        [
                            html.Label("Filter (MongoDB syntax):"),
                            html.Div(
                                [
                                    dcc.Input(
                                        id="filter-input",
                                        type="text",
                                        placeholder='{"age": {"$gt": 20}}',
                                        style={"width": "600px"},
                                    ),
                                    html.Button(
                                        "Apply",
                                        id="apply-filter",
                                        style={"marginLeft": "8px"},
                                    ),
                                ],
                                style={"display": "flex", "alignItems": "center"},
                            ),
                        ],
                        style={"display": "inline-block", "marginRight": "20px"},
                    ),
                    html.Div(
                        [
                            html.Label("Dimensions:"),
                            dcc.RadioItems(
                                id="dim-toggle",
                                options=[
                                    {"label": "2D", "value": 2},
                                    {"label": "3D", "value": 3},
                                ],
                                value=n_components,
                                inline=True,
                            ),
                        ],
                        style={"display": "inline-block"},
                    ),
                ],
                style={"marginBottom": "10px"},
            ),
            html.Div(id="filter-error", style={"color": "red", "marginBottom": "10px"}),
            html.Div(
                id="point-count",
                style={"color": "#666", "fontSize": "14px", "marginBottom": "10px"},
            ),
            dcc.Graph(id="scatter-plot", style={"height": "80vh"}),
        ],
        style={"padding": "20px", "fontFamily": "sans-serif"},
    )

    @app.callback(
        Output("scatter-plot", "figure"),
        Output("filter-error", "children"),
        Output("point-count", "children"),
        Input("apply-filter", "n_clicks"),
        Input("field-dropdown", "value"),
        Input("dim-toggle", "value"),
        State("filter-input", "value"),
    )
    def update_figure(n_clicks, field_path, n_dims, filter_str):
        docs = documents
        error_msg = ""

        if n_dims not in embedding_cache:
            return go.Figure(), f"No {n_dims}D reduction available.", ""

        emb = embedding_cache[n_dims]

        if filter_str:
            try:
                query = json.loads(filter_str)
                docs, mask = filter_documents(docs, query)
                emb = emb[mask]
            except json.JSONDecodeError as e:
                error_msg = f"Invalid JSON: {e}"
            except ValueError as e:
                error_msg = f"Query error: {e}"

        if not field_path or not docs:
            count_msg = f"Showing 0 / {len(documents)} points"
            return go.Figure(), error_msg, count_msg

        count_msg = f"Showing {len(docs)} / {len(documents)} points"

        perspectives = build_perspectives(docs, [field_path])
        hover_text = create_hover_text(docs)

        method_label = method.upper()
        if method_label == "TSNE":
            method_label = "T-SNE"

        fig = create_figure(emb, perspectives, hover_text, method_label=method_label)
        return fig, error_msg, count_msg

    return app
