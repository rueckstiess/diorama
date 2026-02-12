# Diorama

2D/3D embedding visualization for JSON documents. Published on PyPI as `diorama`.

## Toolchain

- **uv** for dependency management and virtual environments
- **ruff** for linting/formatting (line-length 100, py312)
- **pytest** for testing
- **hatchling** build backend, src layout

## Commands

```bash
uv run pytest tests/             # run all 114 tests
uv run ruff check src/ tests/    # lint
uv run ruff format src/ tests/   # format
uv build                         # build sdist + wheel
```

## Package Structure

```
src/diorama/
  __init__.py       # Public API: show(), reduce(), app(), ColoringPerspective
  perspectives.py   # ColoringPerspective dataclass
  fields.py         # Field path extraction, type detection, perspective building
  query.py          # MongoDB query subset matcher (plain Python, no deps)
  hover.py          # JSON hover text formatting for Plotly tooltips
  reduction.py      # UMAP/T-SNE dispatch with lazy imports
  viz.py            # Plotly figure construction (trace-visibility dropdown)
  dashboard.py      # Optional Dash web app
```

## Public API

- `diorama.show(embeddings, documents, ...)` — full pipeline: reduce → filter → color → plot
- `diorama.reduce(embeddings, ...)` — standalone reduction for caching the expensive step
- `diorama.app(embeddings, documents, ...)` — launch interactive Dash web UI

## Key Design Decisions

- **Reduce-then-filter**: ALL embeddings are reduced first, then filtered. This ensures
  stable point positions regardless of which filter is applied.
- **No pandas**: Uses plain lists and numpy arrays instead of DataFrames/Series.
- **Lazy imports**: umap-learn, openTSNE, and dash are imported inside function bodies
  so they're only needed when actually used (optional dependencies).
- **`_MISSING` sentinel** in query.py distinguishes "field absent" from "field is None".
- **Subsample-then-transform**: For large datasets (100K+), fit the reducer on a random
  subset then transform remaining points. Supported for both UMAP and T-SNE (via openTSNE).
- **Plotly hover**: `<pre>` tags don't work in Plotly hover tooltips. Use `<br>` and
  `&nbsp;` for whitespace formatting instead.

## Module Naming Gotcha

Module names must not collide with public API function names. We renamed `reduce.py` →
`reduction.py` and `app.py` → `dashboard.py` because `diorama.reduce` and `diorama.app`
resolved to the modules instead of the functions in `__init__.py`.

## Dependencies

- **Core**: numpy, plotly
- **Optional extras**: `[umap]` (umap-learn, pynndescent), `[tsne]` (openTSNE),
  `[dash]` (dash), `[all]` (everything)
- pynndescent>=0.5.14 floor is required to avoid old numba/llvmlite resolution issues

## Test Structure

Each source module has a corresponding test file (test_query.py, test_fields.py, etc.).
Tests use synthetic data and mock UMAP/T-SNE to stay fast. 114 tests total.
