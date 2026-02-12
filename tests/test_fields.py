"""Tests for field path extraction and type detection."""

from diorama.fields import (
    build_perspectives,
    detect_color_type,
    extract_field_paths,
    extract_values,
    field_coverage,
    top_field_paths,
)


class TestExtractFieldPaths:
    def test_flat_document(self):
        docs = [{"a": 1, "b": 2}]
        assert extract_field_paths(docs) == ["a", "b"]

    def test_nested_document(self):
        docs = [{"a": {"b": 1, "c": 2}}]
        assert extract_field_paths(docs) == ["a.b", "a.c"]

    def test_deeply_nested(self):
        docs = [{"a": {"b": {"c": 3}}}]
        assert extract_field_paths(docs) == ["a.b.c"]

    def test_array_is_leaf(self):
        docs = [{"tags": ["a", "b"], "name": "x"}]
        assert extract_field_paths(docs) == ["name", "tags"]

    def test_union_across_documents(self):
        docs = [{"a": 1}, {"b": 2}, {"a": 3, "c": 4}]
        assert extract_field_paths(docs) == ["a", "b", "c"]

    def test_empty_dict_is_leaf(self):
        docs = [{"a": {}}]
        assert extract_field_paths(docs) == ["a"]

    def test_none_value_is_leaf(self):
        docs = [{"a": None}]
        assert extract_field_paths(docs) == ["a"]

    def test_empty_documents(self):
        assert extract_field_paths([]) == []
        assert extract_field_paths([{}]) == []


class TestExtractValues:
    def test_basic(self):
        docs = [{"x": 1}, {"x": 2}, {"x": 3}]
        assert extract_values(docs, "x") == [1, 2, 3]

    def test_missing_field(self):
        docs = [{"x": 1}, {"y": 2}, {"x": 3}]
        assert extract_values(docs, "x") == [1, None, 3]

    def test_nested(self):
        docs = [{"a": {"b": 10}}, {"a": {"b": 20}}]
        assert extract_values(docs, "a.b") == [10, 20]

    def test_none_value(self):
        docs = [{"x": None}]
        assert extract_values(docs, "x") == [None]


class TestFieldCoverage:
    def test_full_coverage(self):
        docs = [{"a": 1}, {"a": 2}]
        assert field_coverage(docs, "a") == 1.0

    def test_partial_coverage(self):
        docs = [{"a": 1}, {"b": 2}, {"a": 3}, {"b": 4}]
        assert field_coverage(docs, "a") == 0.5

    def test_no_coverage(self):
        docs = [{"b": 1}]
        assert field_coverage(docs, "a") == 0.0

    def test_empty_docs(self):
        assert field_coverage([], "a") == 0.0


class TestDetectColorType:
    def test_all_strings(self):
        assert detect_color_type(["a", "b", "c"]) == "categorical"

    def test_all_bools(self):
        assert detect_color_type([True, False, True]) == "categorical"

    def test_numeric_low_cardinality(self):
        assert detect_color_type([1, 2, 3, 1, 2]) == "categorical"

    def test_numeric_high_cardinality(self):
        values = list(range(50))
        assert detect_color_type(values) == "continuous"

    def test_numeric_at_threshold(self):
        values = list(range(20))
        assert detect_color_type(values, categorical_threshold=20) == "continuous"

    def test_numeric_below_threshold(self):
        values = list(range(19))
        assert detect_color_type(values, categorical_threshold=20) == "categorical"

    def test_mixed_types(self):
        assert detect_color_type([1, "two", 3]) == "categorical"

    def test_all_none(self):
        assert detect_color_type([None, None]) == "categorical"

    def test_empty(self):
        assert detect_color_type([]) == "categorical"

    def test_floats_continuous(self):
        values = [float(i) * 0.1 for i in range(100)]
        assert detect_color_type(values) == "continuous"

    def test_with_nones_mixed_in(self):
        values = list(range(50)) + [None, None]
        assert detect_color_type(values) == "continuous"

    def test_bool_not_treated_as_int(self):
        # Bools are subclass of int in Python - ensure we handle this
        assert detect_color_type([True, False, 0, 1]) == "categorical"


class TestTopFieldPaths:
    def test_sorts_by_coverage(self):
        docs = [
            {"a": 1, "b": 2, "c": 3},
            {"a": 4, "b": 5},
            {"a": 6},
        ]
        paths = top_field_paths(docs)
        assert paths[0] == "a"  # 100% coverage
        assert paths[1] == "b"  # 67% coverage
        assert paths[2] == "c"  # 33% coverage

    def test_caps_at_max(self):
        docs = [{f"field_{i}": i for i in range(30)}]
        paths = top_field_paths(docs, max_paths=5)
        assert len(paths) == 5


class TestBuildPerspectives:
    def test_categorical(self):
        docs = [{"color": "red"}, {"color": "blue"}, {"color": "red"}]
        persp = build_perspectives(docs, ["color"])
        assert len(persp) == 1
        assert persp[0].name == "color"
        assert persp[0].color_type == "categorical"
        assert persp[0].values == ["red", "blue", "red"]

    def test_continuous(self):
        docs = [{"score": float(i)} for i in range(50)]
        persp = build_perspectives(docs, ["score"])
        assert persp[0].color_type == "continuous"
        assert persp[0].values[0] == 0.0

    def test_missing_becomes_na(self):
        docs = [{"a": 1}, {}]
        persp = build_perspectives(docs, ["a"])
        assert persp[0].color_type == "categorical"  # low cardinality
        assert persp[0].values == ["1", "N/A"]

    def test_override(self):
        docs = [{"x": i} for i in range(5)]
        persp = build_perspectives(docs, ["x"], color_type_overrides={"x": "continuous"})
        assert persp[0].color_type == "continuous"

    def test_multiple_paths(self):
        docs = [{"a": "x", "b": 1.0} for _ in range(5)]
        persp = build_perspectives(docs, ["a", "b"])
        assert len(persp) == 2
        assert persp[0].name == "a"
        assert persp[1].name == "b"
