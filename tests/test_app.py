"""Tests for the Dash app module."""

import numpy as np
import pytest

dash = pytest.importorskip("dash")

from diorama.dashboard import create_app  # noqa: E402


@pytest.fixture
def sample_data():
    docs = [
        {"name": f"item_{i}", "category": f"cat_{i % 3}", "score": float(i)}
        for i in range(20)
    ]
    embeddings = np.random.randn(20, 2)  # pre-reduced
    return embeddings, docs


class TestCreateApp:
    def test_returns_dash_app(self, sample_data):
        embeddings, docs = sample_data
        app = create_app(embeddings, docs)
        assert isinstance(app, dash.Dash)

    def test_layout_has_components(self, sample_data):
        embeddings, docs = sample_data
        app = create_app(embeddings, docs)
        layout_str = str(app.layout)
        assert "field-dropdown" in layout_str
        assert "filter-input" in layout_str
        assert "dim-toggle" in layout_str
        assert "scatter-plot" in layout_str
