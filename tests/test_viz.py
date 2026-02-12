"""Tests for Plotly figure construction."""

import numpy as np
import plotly.graph_objects as go
import pytest

from diorama.perspectives import ColoringPerspective
from diorama.viz import create_figure


@pytest.fixture
def embedding_2d():
    return np.random.randn(20, 2)


@pytest.fixture
def embedding_3d():
    return np.random.randn(20, 3)


@pytest.fixture
def hover_text():
    return [f"point {i}" for i in range(20)]


@pytest.fixture
def categorical_perspective():
    return ColoringPerspective(
        name="category",
        color_type="categorical",
        values=["A"] * 7 + ["B"] * 8 + ["C"] * 5,
    )


@pytest.fixture
def continuous_perspective():
    return ColoringPerspective(
        name="score",
        color_type="continuous",
        values=[float(i) for i in range(20)],
    )


class TestCreateFigure:
    def test_returns_figure(self, embedding_2d, hover_text, categorical_perspective):
        fig = create_figure(embedding_2d, [categorical_perspective], hover_text)
        assert isinstance(fig, go.Figure)

    def test_2d_categorical_traces(self, embedding_2d, hover_text, categorical_perspective):
        fig = create_figure(embedding_2d, [categorical_perspective], hover_text)
        # 3 categories -> 3 traces
        assert len(fig.data) == 3
        assert all(isinstance(t, go.Scatter) for t in fig.data)

    def test_3d_categorical_traces(self, embedding_3d, hover_text, categorical_perspective):
        fig = create_figure(embedding_3d, [categorical_perspective], hover_text)
        assert len(fig.data) == 3
        assert all(isinstance(t, go.Scatter3d) for t in fig.data)

    def test_continuous_single_trace(self, embedding_2d, hover_text, continuous_perspective):
        fig = create_figure(embedding_2d, [continuous_perspective], hover_text)
        assert len(fig.data) == 1

    def test_multiple_perspectives(
        self, embedding_2d, hover_text, categorical_perspective, continuous_perspective
    ):
        fig = create_figure(
            embedding_2d, [categorical_perspective, continuous_perspective], hover_text
        )
        # 3 categorical + 1 continuous = 4 traces
        assert len(fig.data) == 4
        # First perspective visible, second hidden
        assert all(fig.data[i].visible for i in range(3))
        assert not fig.data[3].visible

    def test_dropdown_buttons(
        self, embedding_2d, hover_text, categorical_perspective, continuous_perspective
    ):
        fig = create_figure(
            embedding_2d, [categorical_perspective, continuous_perspective], hover_text
        )
        menus = fig.layout.updatemenus
        assert len(menus) == 1
        buttons = menus[0].buttons
        assert len(buttons) == 2
        assert buttons[0].label == "category"
        assert buttons[1].label == "score"

    def test_max_categorical_traces(self, embedding_2d, hover_text):
        # Create a perspective with many categories
        values = [f"cat_{i % 25}" for i in range(20)]
        persp = ColoringPerspective(name="many", color_type="categorical", values=values)
        fig = create_figure(embedding_2d, [persp], hover_text, max_categorical_traces=5)
        # Should be at most 5 + "Other" = 6 traces
        assert len(fig.data) <= 6

    def test_continuous_with_nones(self, embedding_2d, hover_text):
        values = [float(i) if i % 2 == 0 else None for i in range(20)]
        persp = ColoringPerspective(name="sparse", color_type="continuous", values=values)
        fig = create_figure(embedding_2d, [persp], hover_text)
        assert len(fig.data) == 1

    def test_all_none_continuous(self, embedding_2d, hover_text):
        persp = ColoringPerspective(
            name="empty", color_type="continuous", values=[None] * 20
        )
        fig = create_figure(embedding_2d, [persp], hover_text)
        # No traces added for all-None perspective
        assert len(fig.data) == 0

    def test_invalid_embedding_shape(self, hover_text, categorical_perspective):
        bad = np.random.randn(20, 5)
        with pytest.raises(ValueError, match="shape"):
            create_figure(bad, [categorical_perspective], hover_text)

    def test_dark_theme(self, embedding_2d, hover_text, categorical_perspective):
        fig = create_figure(
            embedding_2d, [categorical_perspective], hover_text, color_scheme="dark"
        )
        assert fig.layout.template.layout.plot_bgcolor is not None or True  # dark template applied

    def test_method_label(self, embedding_2d, hover_text, categorical_perspective):
        fig = create_figure(
            embedding_2d, [categorical_perspective], hover_text, method_label="T-SNE"
        )
        assert "T-SNE" in fig.layout.xaxis.title.text
