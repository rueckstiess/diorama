"""Field path extraction, value extraction, and type detection for JSON documents."""

from typing import Any, Literal

from diorama.perspectives import ColoringPerspective
from diorama.query import _MISSING, get_value_at_path


def extract_field_paths(documents: list[dict]) -> list[str]:
    """Extract all unique dot-notation leaf field paths from documents.

    Recurses into nested dicts. Arrays and other non-dict values are treated
    as leaves (not traversed).

    Returns paths sorted alphabetically.
    """
    paths: set[str] = set()
    for doc in documents:
        _walk(doc, "", paths)
    return sorted(paths)


def _walk(obj: dict, prefix: str, paths: set[str]) -> None:
    for key, value in obj.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and value:
            _walk(value, path, paths)
        else:
            paths.add(path)


def extract_values(documents: list[dict], field_path: str) -> list[Any]:
    """Extract the value at field_path from each document.

    Returns None for documents where the field is missing.
    """
    result = []
    for doc in documents:
        val = get_value_at_path(doc, field_path)
        result.append(None if val is _MISSING else val)
    return result


def field_coverage(documents: list[dict], field_path: str) -> float:
    """Return the fraction of documents that have the given field path."""
    count = sum(1 for doc in documents if get_value_at_path(doc, field_path) is not _MISSING)
    return count / len(documents) if documents else 0.0


def detect_color_type(
    values: list[Any],
    categorical_threshold: int = 20,
) -> Literal["categorical", "continuous"]:
    """Determine whether a field should be colored as categorical or continuous.

    Rules:
    1. All non-None booleans -> categorical.
    2. All non-None strings -> categorical.
    3. All non-None int/float (not bool):
       - Fewer than categorical_threshold unique values -> categorical.
       - Otherwise -> continuous.
    4. Mixed types or no values -> categorical.
    """
    non_none = [v for v in values if v is not None]
    if not non_none:
        return "categorical"

    # Check if all values are the same type category
    all_bool = all(isinstance(v, bool) for v in non_none)
    if all_bool:
        return "categorical"

    all_str = all(isinstance(v, str) for v in non_none)
    if all_str:
        return "categorical"

    # Check numeric (excluding bool, since bool is subclass of int)
    all_numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in non_none)
    if all_numeric:
        unique_count = len(set(non_none))
        if unique_count < categorical_threshold:
            return "categorical"
        return "continuous"

    # Mixed types
    return "categorical"


def top_field_paths(
    documents: list[dict],
    max_paths: int = 15,
) -> list[str]:
    """Return the top field paths sorted by coverage, capped at max_paths."""
    all_paths = extract_field_paths(documents)
    paths_with_coverage = [(p, field_coverage(documents, p)) for p in all_paths]
    paths_with_coverage.sort(key=lambda x: (-x[1], x[0]))
    return [p for p, _ in paths_with_coverage[:max_paths]]


def build_perspectives(
    documents: list[dict],
    color_by: list[str],
    *,
    color_type_overrides: dict[str, str] | None = None,
    categorical_threshold: int = 20,
) -> list[ColoringPerspective]:
    """Build ColoringPerspective objects for each field path."""
    perspectives = []
    for path in color_by:
        raw_values = extract_values(documents, path)

        if color_type_overrides and path in color_type_overrides:
            color_type = color_type_overrides[path]
        else:
            color_type = detect_color_type(raw_values, categorical_threshold)

        if color_type == "categorical":
            values = [str(v) if v is not None else "N/A" for v in raw_values]
        else:
            values = [
                float(v)
                if isinstance(v, (int, float)) and not isinstance(v, bool)
                else None
                for v in raw_values
            ]

        perspectives.append(
            ColoringPerspective(
                name=path,
                color_type=color_type,
                values=values,
            )
        )
    return perspectives
