# Diorama

Interactive 2D/3D embedding visualization for JSON documents.

Diorama takes high-dimensional embeddings and a list of JSON documents, runs dimensionality reduction, and produces interactive Plotly scatter plots. Color points by any field in your documents, filter with MongoDB query syntax, and switch between perspectives via dropdown.

## Install

```bash
pip install diorama[umap]        # core + UMAP (recommended)
pip install diorama[all]         # core + UMAP + T-SNE + Dash app
```

## Quick start

```python
import numpy as np
import diorama

# Load your data
documents = [{"city": "Sydney", "score": 4.2, "category": "A"}, ...]
embeddings = np.load("embeddings.npy")  # (N, D) array

# Reduce and visualize in one call
fig = diorama.show(embeddings, documents, color_by="category")
```

## Reduce once, explore fast

UMAP is the expensive step. Run it once and reuse the result:

```python
reduced = diorama.reduce(embeddings, n_components=2, show_progress=True)

# Fast iterations — only filtering + plotting
diorama.show(reduced, documents, color_by="city")
diorama.show(reduced, documents, color_by="score")
diorama.show(reduced, documents, filter={"score": {"$gt": 3.0}})
```

Positions are stable across calls — the same document always appears at the same coordinates.

## Large datasets

For 100K+ points, fit UMAP on a subset and transform the rest:

```python
reduced = diorama.reduce(embeddings, n_components=2, subsample=10_000, show_progress=True)
```

## Coloring

Pass a single field, a list of fields (creates a dropdown), or `None` to auto-discover:

```python
diorama.show(reduced, documents, color_by="category")                  # single field
diorama.show(reduced, documents, color_by=["category", "city", "score"])  # dropdown
diorama.show(reduced, documents)                                        # auto-discover top 15
```

Numeric fields with many unique values are colored continuously. Low-cardinality numerics, strings, and booleans are categorical. Override with:

```python
diorama.show(reduced, documents, color_by="score", color_type_overrides={"score": "categorical"})
```

## Filtering

Filter documents using MongoDB query syntax:

```python
diorama.show(reduced, documents, filter={"score": {"$gte": 4.0}, "city": "Sydney"})
diorama.show(reduced, documents, filter={"$or": [{"city": "Sydney"}, {"city": "Melbourne"}]})
diorama.show(reduced, documents, filter={"name": {"$regex": "^A"}})
```

Supported operators: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$exists`, `$regex`, `$and`, `$or`, `$not`, `$nor`.

## 3D

```python
reduced_3d = diorama.reduce(embeddings, n_components=3)
diorama.show(reduced_3d, documents, color_by="category")
```

## Dash app

Launch an interactive web UI with a field dropdown, filter text input, and 2D/3D toggle:

```python
diorama.app(embeddings, documents, subsample=10_000)
# Opens at http://127.0.0.1:8050
```

Requires the `dash` extra: `pip install diorama[dash]`.

## Options

```python
diorama.show(
    embeddings,
    documents,
    color_by="field.path",          # dot notation for nested fields
    filter={"age": {"$gt": 20}},    # MongoDB-style filter
    n_components=2,                  # 2 or 3
    method="umap",                   # "umap" or "tsne"
    subsample=10_000,                # fit on subset, transform all
    color_scheme="dark",             # "light" or "dark"
    height=800,                      # figure height in pixels
    show_progress=True,              # tqdm progress bar
    output_path="plot.html",         # save to file
)
```
