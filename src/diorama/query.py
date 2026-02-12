"""MongoDB query subset matcher for filtering JSON documents.

Evaluates MongoDB-style queries against plain Python dicts. Supports:
- Comparison: $eq (implicit), $ne, $gt, $gte, $lt, $lte
- Array: $in, $nin
- Logical: $and (implicit top-level), $or, $not, $nor
- Element: $exists
- String: $regex

Dot-notation field paths traverse nested dicts (e.g. "address.city").
"""

import re
from typing import Any

import numpy as np

_MISSING = object()


def get_value_at_path(doc: dict, path: str) -> Any:
    """Traverse a nested dict by dot-separated path.

    Returns _MISSING sentinel if any segment is not found.
    """
    current = doc
    for segment in path.split("."):
        if not isinstance(current, dict) or segment not in current:
            return _MISSING
        current = current[segment]
    return current


def match(document: dict, query: dict) -> bool:
    """Return True if document matches the MongoDB-style query."""
    for key, condition in query.items():
        if key == "$and":
            if not all(match(document, sub) for sub in condition):
                return False
        elif key == "$or":
            if not any(match(document, sub) for sub in condition):
                return False
        elif key == "$nor":
            if any(match(document, sub) for sub in condition):
                return False
        else:
            value = get_value_at_path(document, key)
            if not _match_condition(value, condition):
                return False
    return True


def _match_condition(value: Any, condition: Any) -> bool:
    """Match a value against a condition.

    condition can be:
    - A scalar (implicit $eq)
    - A dict of operators: {"$gt": 5, "$lt": 10}
    """
    if not isinstance(condition, dict):
        return value == condition

    for op, operand in condition.items():
        if op == "$eq":
            if value != operand:
                return False
        elif op == "$ne":
            if value == operand:
                return False
        elif op == "$gt":
            if value is _MISSING or value is None or value <= operand:
                return False
        elif op == "$gte":
            if value is _MISSING or value is None or value < operand:
                return False
        elif op == "$lt":
            if value is _MISSING or value is None or value >= operand:
                return False
        elif op == "$lte":
            if value is _MISSING or value is None or value > operand:
                return False
        elif op == "$in":
            if value is _MISSING or value not in operand:
                return False
        elif op == "$nin":
            if value is not _MISSING and value in operand:
                return False
        elif op == "$exists":
            exists = value is not _MISSING
            if exists != operand:
                return False
        elif op == "$regex":
            if not isinstance(value, str) or not re.search(operand, value):
                return False
        elif op == "$not":
            if _match_condition(value, operand):
                return False
        else:
            raise ValueError(f"Unsupported query operator: {op}")
    return True


def filter_documents(
    documents: list[dict],
    query: dict,
) -> tuple[list[dict], np.ndarray]:
    """Filter documents by a MongoDB-style query.

    Returns:
        Tuple of (filtered_documents, boolean_mask). The mask can be used
        to index into the corresponding embeddings array.
    """
    mask = np.array([match(doc, query) for doc in documents])
    filtered = [doc for doc, m in zip(documents, mask) if m]
    return filtered, mask
