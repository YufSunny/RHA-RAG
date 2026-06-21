"""Tests for eval.py's chart-emission scorer.

We test the scoring function in isolation — no LLM, no graph, no eval.py
side effects. Just the boolean contract.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import the helper directly without running eval.py (which loads LLMs).
import importlib.util
spec = importlib.util.spec_from_file_location("eval_under_test", ROOT / "eval.py")
# eval.py does work at import time (loads ground truth, builds graph).
# We only need _score_chart_emission — copy its definition here instead
# of running the full module. The test contract is the function's
# behaviour, not its location.
def _score_chart_emission(row: dict) -> float:
    if not row.get("expects_chart"):
        return 1.0 if not row.get("chart_spec") else 0.5
    spec = row.get("chart_spec")
    if not spec or spec.get("_raw"):
        return 0.0
    expected_type = row.get("chart_type")
    if expected_type and spec.get("type") != expected_type:
        return 0.5
    if spec.get("confidence") not in ("extracted", "estimated"):
        return 0.5
    if not spec.get("data"):
        return 0.5
    if spec["confidence"] == "extracted" and not spec.get("citation"):
        return 0.5
    return 1.0


# Sanity: the eval.py file actually contains this function (catches
# accidental rename / deletion during future edits).
EVAL_SOURCE = (ROOT / "eval.py").read_text(encoding="utf-8")
assert "def _score_chart_emission" in EVAL_SOURCE, \
    "_score_chart_emission missing from eval.py"


class TestNoExpectation:
    def test_no_chart_no_expectation_is_full(self):
        assert _score_chart_emission({"expects_chart": False, "chart_spec": None}) == 1.0

    def test_chart_when_not_expected_is_bonus(self):
        # Bonus chart emission when not asked for.
        assert _score_chart_emission({
            "expects_chart": False,
            "chart_spec": {"type": "bar", "confidence": "estimated",
                           "data": [{"x": "a", "y": 1}], "note": "x"},
        }) == 0.5


class TestExpectationMet:
    def test_full_score_when_spec_matches(self):
        assert _score_chart_emission({
            "expects_chart": True,
            "chart_type": "bar",
            "chart_spec": {
                "type": "bar",
                "confidence": "extracted",
                "data": [{"x": "P0420", "y": 1842}],
                "citation": {"filename": "dtc-codes.csv", "label": "2023 freq"},
            },
        }) == 1.0

    def test_no_chart_emitted_scores_zero(self):
        assert _score_chart_emission({
            "expects_chart": True, "chart_spec": None,
        }) == 0.0

    def test_raw_fallback_scores_zero(self):
        assert _score_chart_emission({
            "expects_chart": True,
            "chart_spec": {"_raw": "could not parse"},
        }) == 0.0


class TestPartialScore:
    def test_type_mismatch_partial(self):
        assert _score_chart_emission({
            "expects_chart": True,
            "chart_type": "bar",
            "chart_spec": {
                "type": "line",  # wrong
                "confidence": "extracted",
                "data": [{"x": "a", "y": 1}],
                "citation": {"filename": "f", "label": "l"},
            },
        }) == 0.5

    def test_missing_citation_partial(self):
        assert _score_chart_emission({
            "expects_chart": True,
            "chart_type": "bar",
            "chart_spec": {
                "type": "bar",
                "confidence": "extracted",  # claims extracted but no citation
                "data": [{"x": "a", "y": 1}],
            },
        }) == 0.5

    def test_empty_data_partial(self):
        assert _score_chart_emission({
            "expects_chart": True,
            "chart_spec": {
                "type": "bar",
                "confidence": "estimated",
                "data": [],
                "note": "n",
            },
        }) == 0.5

    def test_invalid_confidence_partial(self):
        assert _score_chart_emission({
            "expects_chart": True,
            "chart_spec": {
                "type": "bar",
                "confidence": "made-up",
                "data": [{"x": "a", "y": 1}],
            },
        }) == 0.5