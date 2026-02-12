"""Dimensionality reduction dispatch for UMAP and T-SNE."""

import numpy as np


def reduce_embeddings(
    embeddings: np.ndarray,
    *,
    n_components: int = 2,
    method: str = "umap",
    show_progress: bool = False,
    subsample: int | None = None,
    **kwargs,
) -> np.ndarray:
    """Reduce high-dimensional embeddings to 2D or 3D.

    If embeddings already have n_components dimensions, returns them unchanged.

    Args:
        embeddings: (N, D) array.
        n_components: Target dimensions (2 or 3).
        method: "umap" or "tsne".
        show_progress: Show progress bar during reduction.
        subsample: If set, fit the reducer on this many random points, then
            transform all points. Much faster for large datasets. Only
            supported for UMAP.
        **kwargs: Passed to the underlying reducer constructor.

    Returns:
        (N, n_components) numpy array.
    """
    if n_components not in (2, 3):
        raise ValueError(f"n_components must be 2 or 3, got {n_components}")

    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D array, got shape {embeddings.shape}")

    d = embeddings.shape[1]

    if d == n_components:
        return embeddings

    if d < n_components:
        raise ValueError(
            f"Cannot reduce {d}-dimensional embeddings to {n_components} dimensions. "
            f"Input must have more dimensions than n_components."
        )

    if method == "umap":
        return _reduce_umap(
            embeddings, n_components, show_progress=show_progress, subsample=subsample, **kwargs
        )
    elif method == "tsne":
        if subsample is not None:
            raise ValueError("subsample is not supported for T-SNE (no transform method).")
        return _reduce_tsne(embeddings, n_components, show_progress=show_progress, **kwargs)
    else:
        raise ValueError(f"Unknown reduction method: {method!r}. Supported: 'umap', 'tsne'.")


def _reduce_umap(
    embeddings: np.ndarray,
    n_components: int,
    *,
    show_progress: bool = False,
    subsample: int | None = None,
    **kwargs,
) -> np.ndarray:
    try:
        from umap import UMAP
    except ImportError:
        raise ImportError(
            "umap-learn is required for UMAP reduction. Install it with: pip install diorama[umap]"
        ) from None

    defaults = {"n_neighbors": 30, "min_dist": 0.05, "metric": "cosine", "low_memory": False}
    params = {**defaults, **kwargs, "n_components": n_components, "verbose": show_progress}
    reducer = UMAP(**params)

    n = len(embeddings)
    if subsample is not None and subsample < n:
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(n, size=subsample, replace=False)
        reducer.fit(embeddings[sample_idx])
        return reducer.transform(embeddings)
    else:
        return reducer.fit_transform(embeddings)


def _reduce_tsne(
    embeddings: np.ndarray, n_components: int, *, show_progress: bool = False, **kwargs
) -> np.ndarray:
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        raise ImportError(
            "scikit-learn is required for T-SNE reduction. "
            "Install it with: pip install diorama[tsne]"
        ) from None

    params = {**kwargs, "n_components": n_components, "verbose": 1 if show_progress else 0}
    reducer = TSNE(**params)
    return reducer.fit_transform(embeddings)
