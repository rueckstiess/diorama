"""Tests for hover text generation."""

from diorama.hover import create_hover_text


class TestCreateHoverText:
    def test_basic(self):
        docs = [{"name": "alice", "age": 30}]
        result = create_hover_text(docs)
        assert len(result) == 1
        assert "alice" in result[0]
        assert "30" in result[0]
        assert "<br>" in result[0]
        assert "&nbsp;&nbsp;" in result[0]

    def test_truncation(self):
        doc = {f"key_{i}": f"value_{i}" for i in range(100)}
        result = create_hover_text([doc], max_length=50)
        assert "..." in result[0]

    def test_no_truncation_for_small_doc(self):
        result = create_hover_text([{"a": 1}])
        assert "..." not in result[0]

    def test_multiple_docs(self):
        docs = [{"a": 1}, {"b": 2}, {"c": 3}]
        result = create_hover_text(docs)
        assert len(result) == 3

    def test_nested_doc(self):
        docs = [{"user": {"name": "bob", "address": {"city": "Sydney"}}}]
        result = create_hover_text(docs)
        assert "Sydney" in result[0]
