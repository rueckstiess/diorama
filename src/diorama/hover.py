"""Hover text generation for JSON document visualizations."""

import json


def create_hover_text(
    documents: list[dict],
    *,
    max_length: int = 500,
) -> list[str]:
    """Generate HTML hover text for each document.

    Pretty-prints the JSON document, truncating if it exceeds max_length characters.
    """
    hover_texts = []
    for doc in documents:
        text = json.dumps(doc, indent=2, default=str)
        if len(text) > max_length:
            text = text[:max_length] + "\n..."
        # Plotly hover tooltips ignore <pre> whitespace — use <br> and &nbsp;
        html = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html = html.replace("\n", "<br>").replace("  ", "&nbsp;&nbsp;")
        hover_texts.append(html)
    return hover_texts
