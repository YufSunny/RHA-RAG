"""Tests for the ChartSpec Pydantic schema.

Covers:
- Valid spec with `extracted` confidence accepts a citation.
- Valid spec with `estimated` confidence accepts a note.
- `extracted` without a citation fails.
- `estimated` without a note fails.
- Unknown chart type fails.
- Empty data array fails (must contain at least one point).
- String fields with HTML/JS render as plain text (no XSS) — verified
  here by checking that Pydantic stores them as strings, not interpreted.
"""

import pytest
from pydantic import ValidationError

from rha_rag.viz import ChartSpec, validate_chart_spec


class TestAcceptance:
    def test_extracted_with_citation(self, sample_chart_spec):
        spec = validate_chart_spec(sample_chart_spec)
        assert spec.type == "bar"
        assert spec.confidence == "extracted"
        assert spec.citation.filename == "dtc-codes.csv"

    def test_estimated_with_note(self, sample_estimated_chart_spec):
        spec = validate_chart_spec(sample_estimated_chart_spec)
        assert spec.type == "line"
        assert spec.confidence == "estimated"
        assert "Aggregated" in spec.note

    def test_all_chart_types_accepted(self):
        for t in ("bar", "line", "pie", "scatter", "table"):
            spec = validate_chart_spec({
                "type": t, "title": "x", "y_label": "y",
                "data": [{"x": "a", "y": 1}], "confidence": "estimated",
                "note": "n",
            })
            assert spec.type == t

    def test_optional_label_on_data_point(self, sample_chart_spec):
        spec = validate_chart_spec(sample_chart_spec)
        assert spec.data[0].label == "Catalyst Bank 1"
        assert spec.data[1].label == "Catalyst Bank 2"


class TestRejection:
    def test_extracted_without_citation_fails(self, sample_chart_spec):
        bad = {**sample_chart_spec}
        bad["confidence"] = "extracted"
        bad.pop("citation", None)
        with pytest.raises(ValidationError, match="requires a citation"):
            validate_chart_spec(bad)

    def test_estimated_without_note_fails(self, sample_estimated_chart_spec):
        bad = {**sample_estimated_chart_spec}
        bad.pop("note", None)
        with pytest.raises(ValidationError, match="requires a note"):
            validate_chart_spec(bad)

    def test_unknown_chart_type_fails(self, sample_chart_spec):
        bad = {**sample_chart_spec, "type": "radar"}
        with pytest.raises(ValidationError):
            validate_chart_spec(bad)

    def test_empty_data_fails(self, sample_chart_spec):
        bad = {**sample_chart_spec, "data": []}
        with pytest.raises(ValidationError):
            validate_chart_spec(bad)

    def test_y_value_must_be_numeric(self, sample_chart_spec):
        bad = {**sample_chart_spec, "data": [{"x": "a", "y": "not a number"}]}
        with pytest.raises(ValidationError):
            validate_chart_spec(bad)

    def test_unknown_confidence_fails(self, sample_chart_spec):
        bad = {**sample_chart_spec, "confidence": "made-up"}
        with pytest.raises(ValidationError):
            validate_chart_spec(bad)


class TestStringFields:
    """The frontend contract: string fields are rendered as TEXT. They
    must be stored as plain strings — Pydantic does no HTML parsing, so
    the spec itself can't smuggle executable content."""

    def test_html_in_title_stays_string(self, sample_chart_spec):
        bad = {**sample_chart_spec, "title": "<script>alert(1)</script>"}
        spec = validate_chart_spec(bad)
        assert spec.title == "<script>alert(1)</script>"
        assert isinstance(spec.title, str)

    def test_html_in_label_stays_string(self, sample_chart_spec):
        bad = {**sample_chart_spec, "x_label": "<img src=x onerror=alert(1)>"}
        spec = validate_chart_spec(bad)
        assert spec.x_label == "<img src=x onerror=alert(1)>"

    def test_json_in_note_stays_string(self, sample_chart_spec):
        bad = {**sample_chart_spec, "confidence": "estimated",
               "note": "{'fn': '() => fetch(\"/api/clear\")'}"}
        spec = validate_chart_spec(bad)
        assert spec.note.startswith("{'fn'")


class TestRoundTrip:
    def test_dump_and_reload(self, sample_chart_spec):
        spec = validate_chart_spec(sample_chart_spec)
        roundtripped = validate_chart_spec(spec.model_dump())
        assert roundtripped == spec

    def test_json_serializable(self, sample_chart_spec):
        import json
        spec = validate_chart_spec(sample_chart_spec)
        # Must serialize to plain JSON (no Python-only types).
        json.dumps(spec.model_dump())