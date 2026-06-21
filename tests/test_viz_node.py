"""Tests for the visualize node + graph wiring.

Covers:
- `build_graph` includes the `visualize` node (8 nodes total).
- Both fast and full routes reach `generate_answer -> visualize -> END`.
- Stubbed LLM emitting a valid JSON spec → visualize node produces a
  ChartSpec message in state.messages.
- Stubbed LLM emitting garbage → visualize node still produces a
  ChartSpec message with confidence='estimated' (graceful degradation).
"""

from pathlib import Path
import json
import sys

import pytest

# Mirror the project's load order: rha_rag imports must work, which
# requires the project root on sys.path.
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

from rha_rag.graph import RhaState, build_graph  # noqa: E402
from rha_rag.viz import ChartSpec, validate_chart_spec  # noqa: E402


def _spec_from_message(msg):
    """The visualize node stores the spec as JSON in `.content`."""
    return validate_chart_spec(json.loads(msg.content))


class _StubLLM:
    """Minimal LLM stub: configurable content + tool_calls."""

    def __init__(self, content="", tool_calls=None):
        self._content = content
        self._tool_calls = tool_calls

    def invoke(self, *args, **kwargs):
        return AIMessage(content=self._content, tool_calls=self._tool_calls or [])

    def bind_tools(self, tools):
        return self


VALID_SPEC_JSON = """{
  "type": "bar",
  "title": "P0420 frequency",
  "x_label": "Code",
  "y_label": "Count",
  "data": [
    {"x": "P0420", "y": 1842, "label": "Catalyst Bank 1"},
    {"x": "P0171", "y": 1503, "label": "System Lean Bank 1"}
  ],
  "citation": {"filename": "dtc-codes.csv", "label": "OBD-II frequency 2023"},
  "confidence": "extracted"
}"""


GARBAGE_SPEC = "I cannot emit a chart right now."  # no JSON object


class TestGraphTopology:
    def test_visualize_node_registered(self):
        g = build_graph(None, _StubLLM(), _StubLLM())
        assert "visualize" in g.nodes

    def test_eight_logical_nodes(self):
        # LangGraph adds an implicit "__start__" entry marker, so the
        # total is 9 — but 8 of them are the named pipeline nodes.
        g = build_graph(None, _StubLLM(), _StubLLM())
        named = [n for n in g.nodes.keys() if not n.startswith("__")]
        assert len(named) == 8

    def test_full_path_includes_all_nodes(self):
        g = build_graph(None, _StubLLM(), _StubLLM())
        expected = {"clarify", "generate_query", "retrieve", "grade",
                    "reason", "verify", "generate_answer", "visualize"}
        actual = {n for n in g.nodes.keys() if not n.startswith("__")}
        assert actual == expected


class TestVisualizeNodeFactory:
    def _make_state(self, llm_content=""):
        return RhaState(
            question="What is P0420?",
            history="",
            fast_mode=False,
            visual_spec="",
            messages=[
                HumanMessage(content="What is P0420?"),
                AIMessage(content="P0420 = Catalyst efficiency below threshold."),
                ToolMessage(
                    content="[Source: dtc-codes.csv]: P0420 ...",
                    tool_call_id="stub-tool-call-1",
                ),
            ],
        )

    def test_valid_json_emits_valid_chart_spec(self):
        from rha_rag.viz import make_visualize
        node = make_visualize(_StubLLM(content=VALID_SPEC_JSON))
        out = node(self._make_state())
        # The node returns a messages update; the last message's content
        # is the serialized ChartSpec.
        spec = _spec_from_message(out["messages"][-1])
        assert spec.type == "bar"
        assert spec.confidence == "extracted"
        assert spec.citation.filename == "dtc-codes.csv"
        assert len(spec.data) == 2

    def test_garbage_output_falls_back_to_estimated(self):
        from rha_rag.viz import make_visualize
        node = make_visualize(_StubLLM(content=GARBAGE_SPEC))
        out = node(self._make_state())
        spec = _spec_from_message(out["messages"][-1])
        # Graceful degradation: confidence='estimated' + a `note`.
        assert spec.confidence == "estimated"
        assert spec.note  # non-empty

    def test_missing_citation_downgrades_to_estimated(self):
        from rha_rag.viz import make_visualize
        # LLM claims extracted but no citation in spec → validation
        # fails → node emits a fallback spec.
        bad = """{
          "type": "bar", "title": "x", "y_label": "y",
          "data": [{"x": "a", "y": 1}],
          "confidence": "extracted"
        }"""
        node = make_visualize(_StubLLM(content=bad))
        out = node(self._make_state())
        spec = _spec_from_message(out["messages"][-1])
        # Fallback is always 'estimated'.
        assert spec.confidence == "estimated"

    def test_empty_state_uses_question_as_answer(self):
        from rha_rag.viz import make_visualize
        # No AI message in state — node should fall back to the
        # question field for the prompt.
        state = RhaState(
            question="any question",
            history="",
            fast_mode=False,
            visual_spec="",
            messages=[HumanMessage(content="any question")],
        )
        node = make_visualize(_StubLLM(content=VALID_SPEC_JSON))
        out = node(state)
        spec = _spec_from_message(out["messages"][-1])
        assert spec.confidence == "extracted"