"""Tests for dimensionality reduction dispatch."""

import numpy as np
import pytest

from diorama.reduction import reduce_embeddings


class TestReduceEmbeddings:
    def test_passthrough_2d(self):
        data = np.random.randn(10, 2)
        result = reduce_embeddings(data, n_components=2)
        np.testing.assert_array_equal(result, data)

    def test_passthrough_3d(self):
        data = np.random.randn(10, 3)
        result = reduce_embeddings(data, n_components=3)
        np.testing.assert_array_equal(result, data)

    def test_invalid_n_components(self):
        data = np.random.randn(10, 5)
        with pytest.raises(ValueError, match="n_components must be 2 or 3"):
            reduce_embeddings(data, n_components=4)

    def test_1d_input_raises(self):
        data = np.random.randn(10)
        with pytest.raises(ValueError, match="must be 2D array"):
            reduce_embeddings(data, n_components=2)

    def test_cannot_reduce_lower_dim(self):
        data = np.random.randn(10, 2)
        with pytest.raises(ValueError, match="Cannot reduce"):
            reduce_embeddings(data, n_components=3)

    def test_unknown_method(self):
        data = np.random.randn(10, 5)
        with pytest.raises(ValueError, match="Unknown reduction method"):
            reduce_embeddings(data, method="pca")

    def test_umap_reduction(self):
        pytest.importorskip("umap")
        data = np.random.randn(50, 10)
        result = reduce_embeddings(data, n_components=2, method="umap")
        assert result.shape == (50, 2)

    def test_umap_3d(self):
        pytest.importorskip("umap")
        data = np.random.randn(50, 10)
        result = reduce_embeddings(data, n_components=3, method="umap")
        assert result.shape == (50, 3)

    def test_tsne_reduction(self):
        pytest.importorskip("openTSNE")
        data = np.random.randn(50, 10)
        result = reduce_embeddings(data, n_components=2, method="tsne")
        assert result.shape == (50, 2)

    def test_tsne_3d(self):
        pytest.importorskip("openTSNE")
        data = np.random.randn(50, 10)
        result = reduce_embeddings(data, n_components=3, method="tsne")
        assert result.shape == (50, 3)

    def test_tsne_subsample(self):
        pytest.importorskip("openTSNE")
        data = np.random.randn(100, 10)
        result = reduce_embeddings(data, n_components=2, method="tsne", subsample=30)
        assert result.shape == (100, 2)
