"""Integration tests for the diorama.show() public API."""

import numpy as np
import plotly.graph_objects as go
import pytest

import diorama


@pytest.fixture
def sample_docs():
    return [
        {"name": f"item_{i}", "category": f"cat_{i % 5}", "score": float(i), "active": i % 2 == 0}
        for i in range(30)
    ]


@pytest.fixture
def pre_reduced_2d():
    return np.random.randn(30, 2)


@pytest.fixture
def pre_reduced_3d():
    return np.random.randn(30, 3)


class TestShow:
    def test_basic_2d(self, pre_reduced_2d, sample_docs):
        fig = diorama.show(pre_reduced_2d, sample_docs, color_by="category")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_basic_3d(self, pre_reduced_3d, sample_docs):
        fig = diorama.show(pre_reduced_3d, sample_docs, color_by="category")
        assert isinstance(fig, go.Figure)
        assert all(isinstance(t, go.Scatter3d) for t in fig.data)

    def test_auto_discover_fields(self, pre_reduced_2d, sample_docs):
        fig = diorama.show(pre_reduced_2d, sample_docs)
        assert isinstance(fig, go.Figure)
        # Should have dropdown with multiple perspectives
        assert len(fig.layout.updatemenus) > 0

    def test_multiple_color_by(self, pre_reduced_2d, sample_docs):
        fig = diorama.show(pre_reduced_2d, sample_docs, color_by=["category", "score"])
        menus = fig.layout.updatemenus
        assert len(menus) == 1
        assert len(menus[0].buttons) == 2

    def test_with_filter(self, pre_reduced_2d, sample_docs):
        fig = diorama.show(
            pre_reduced_2d, sample_docs, color_by="category", filter={"score": {"$gte": 20}}
        )
        assert isinstance(fig, go.Figure)
        # Should have fewer points than unfiltered
        total_points = sum(len(t.x) for t in fig.data)
        assert total_points == 10  # items 20-29

    def test_filter_no_matches(self, pre_reduced_2d, sample_docs):
        fig = diorama.show(
            pre_reduced_2d, sample_docs, color_by="category", filter={"score": {"$gt": 999}}
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_continuous_field(self, pre_reduced_2d, sample_docs):
        fig = diorama.show(pre_reduced_2d, sample_docs, color_by="score")
        assert isinstance(fig, go.Figure)
        # score has 30 unique float values -> continuous -> 1 trace
        assert len(fig.data) == 1

    def test_color_type_override(self, pre_reduced_2d, sample_docs):
        fig = diorama.show(
            pre_reduced_2d,
            sample_docs,
            color_by="score",
            color_type_overrides={"score": "categorical"},
        )
        # Override to categorical -> multiple traces
        assert len(fig.data) > 1

    def test_dark_scheme(self, pre_reduced_2d, sample_docs):
        fig = diorama.show(pre_reduced_2d, sample_docs, color_by="category", color_scheme="dark")
        assert isinstance(fig, go.Figure)

    def test_length_mismatch(self, sample_docs):
        bad_embeddings = np.random.randn(10, 2)
        with pytest.raises(ValueError, match="must match"):
            diorama.show(bad_embeddings, sample_docs)

    def test_output_path(self, pre_reduced_2d, sample_docs, tmp_path):
        output = str(tmp_path / "test.html")
        fig = diorama.show(pre_reduced_2d, sample_docs, color_by="category", output_path=output)
        assert isinstance(fig, go.Figure)
        assert (tmp_path / "test.html").exists()


class TestReduce:
    def test_passthrough(self):
        data = np.random.randn(10, 2)
        result = diorama.reduce(data, n_components=2)
        np.testing.assert_array_equal(result, data)


class TestColoringPerspective:
    def test_exported(self):
        p = diorama.ColoringPerspective(name="test", color_type="categorical", values=["a", "b"])
        assert p.name == "test"
