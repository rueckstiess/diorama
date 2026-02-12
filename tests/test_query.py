"""Tests for the MongoDB query matcher."""

import numpy as np
import pytest

from diorama.query import filter_documents, get_value_at_path, match


class TestGetValueAtPath:
    def test_top_level(self):
        assert get_value_at_path({"a": 1}, "a") == 1

    def test_nested(self):
        assert get_value_at_path({"a": {"b": {"c": 3}}}, "a.b.c") == 3

    def test_missing_key(self):
        from diorama.query import _MISSING

        assert get_value_at_path({"a": 1}, "b") is _MISSING

    def test_missing_nested(self):
        from diorama.query import _MISSING

        assert get_value_at_path({"a": {"b": 1}}, "a.c") is _MISSING

    def test_none_value(self):
        assert get_value_at_path({"a": None}, "a") is None

    def test_traverse_through_non_dict(self):
        from diorama.query import _MISSING

        assert get_value_at_path({"a": 42}, "a.b") is _MISSING


class TestImplicitEq:
    def test_string_match(self):
        assert match({"name": "alice"}, {"name": "alice"})

    def test_string_no_match(self):
        assert not match({"name": "alice"}, {"name": "bob"})

    def test_int_match(self):
        assert match({"age": 30}, {"age": 30})

    def test_nested_field(self):
        assert match({"address": {"city": "Sydney"}}, {"address.city": "Sydney"})

    def test_missing_field(self):
        assert not match({"name": "alice"}, {"age": 30})

    def test_multiple_conditions(self):
        doc = {"name": "alice", "age": 30}
        assert match(doc, {"name": "alice", "age": 30})
        assert not match(doc, {"name": "alice", "age": 25})


class TestComparisonOperators:
    def test_eq(self):
        assert match({"x": 5}, {"x": {"$eq": 5}})
        assert not match({"x": 5}, {"x": {"$eq": 6}})

    def test_ne(self):
        assert match({"x": 5}, {"x": {"$ne": 6}})
        assert not match({"x": 5}, {"x": {"$ne": 5}})

    def test_gt(self):
        assert match({"x": 10}, {"x": {"$gt": 5}})
        assert not match({"x": 5}, {"x": {"$gt": 5}})
        assert not match({"x": 3}, {"x": {"$gt": 5}})

    def test_gte(self):
        assert match({"x": 5}, {"x": {"$gte": 5}})
        assert match({"x": 10}, {"x": {"$gte": 5}})
        assert not match({"x": 3}, {"x": {"$gte": 5}})

    def test_lt(self):
        assert match({"x": 3}, {"x": {"$lt": 5}})
        assert not match({"x": 5}, {"x": {"$lt": 5}})

    def test_lte(self):
        assert match({"x": 5}, {"x": {"$lte": 5}})
        assert not match({"x": 6}, {"x": {"$lte": 5}})

    def test_combined_range(self):
        assert match({"x": 5}, {"x": {"$gte": 3, "$lt": 10}})
        assert not match({"x": 2}, {"x": {"$gte": 3, "$lt": 10}})
        assert not match({"x": 10}, {"x": {"$gte": 3, "$lt": 10}})

    def test_comparison_on_missing_field(self):
        assert not match({}, {"x": {"$gt": 5}})

    def test_comparison_on_none_value(self):
        assert not match({"x": None}, {"x": {"$gt": 5}})


class TestArrayOperators:
    def test_in(self):
        assert match({"status": "active"}, {"status": {"$in": ["active", "pending"]}})
        assert not match({"status": "deleted"}, {"status": {"$in": ["active", "pending"]}})

    def test_in_missing_field(self):
        assert not match({}, {"status": {"$in": ["active"]}})

    def test_nin(self):
        assert match({"status": "active"}, {"status": {"$nin": ["deleted", "archived"]}})
        assert not match({"status": "deleted"}, {"status": {"$nin": ["deleted", "archived"]}})

    def test_nin_missing_field(self):
        assert match({}, {"status": {"$nin": ["active"]}})


class TestLogicalOperators:
    def test_and(self):
        doc = {"age": 30, "name": "alice"}
        assert match(doc, {"$and": [{"age": {"$gte": 20}}, {"name": "alice"}]})
        assert not match(doc, {"$and": [{"age": {"$gte": 40}}, {"name": "alice"}]})

    def test_or(self):
        assert match({"age": 15}, {"$or": [{"age": {"$lt": 18}}, {"age": {"$gte": 65}}]})
        assert match({"age": 70}, {"$or": [{"age": {"$lt": 18}}, {"age": {"$gte": 65}}]})
        assert not match({"age": 30}, {"$or": [{"age": {"$lt": 18}}, {"age": {"$gte": 65}}]})

    def test_nor(self):
        assert match({"age": 30}, {"$nor": [{"age": {"$lt": 18}}, {"age": {"$gte": 65}}]})
        assert not match({"age": 15}, {"$nor": [{"age": {"$lt": 18}}, {"age": {"$gte": 65}}]})

    def test_not(self):
        assert match({"x": 3}, {"x": {"$not": {"$gte": 5}}})
        assert not match({"x": 10}, {"x": {"$not": {"$gte": 5}}})


class TestExists:
    def test_exists_true(self):
        assert match({"a": 1}, {"a": {"$exists": True}})
        assert not match({}, {"a": {"$exists": True}})

    def test_exists_false(self):
        assert match({}, {"a": {"$exists": False}})
        assert not match({"a": 1}, {"a": {"$exists": False}})

    def test_exists_with_none_value(self):
        assert match({"a": None}, {"a": {"$exists": True}})


class TestRegex:
    def test_regex_match(self):
        assert match({"name": "alice"}, {"name": {"$regex": "^ali"}})
        assert not match({"name": "bob"}, {"name": {"$regex": "^ali"}})

    def test_regex_non_string(self):
        assert not match({"x": 42}, {"x": {"$regex": "42"}})


class TestUnsupportedOperator:
    def test_raises(self):
        with pytest.raises(ValueError, match="Unsupported query operator"):
            match({"x": 1}, {"x": {"$type": "int"}})


class TestFilterDocuments:
    def test_basic_filter(self):
        docs = [{"age": 10}, {"age": 20}, {"age": 30}, {"age": 40}]
        filtered, mask = filter_documents(docs, {"age": {"$gte": 25}})
        assert filtered == [{"age": 30}, {"age": 40}]
        np.testing.assert_array_equal(mask, [False, False, True, True])

    def test_empty_query(self):
        docs = [{"a": 1}, {"b": 2}]
        filtered, mask = filter_documents(docs, {})
        assert filtered == docs
        np.testing.assert_array_equal(mask, [True, True])

    def test_no_matches(self):
        docs = [{"a": 1}, {"a": 2}]
        filtered, mask = filter_documents(docs, {"a": 99})
        assert filtered == []
        np.testing.assert_array_equal(mask, [False, False])
