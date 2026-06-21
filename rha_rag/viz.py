"""Visualization — ChartSpec schema and node factory.

The visualization contract: the `make_visualize` node reads the
generate_answer output and the retrieved context, asks the LLM for a
JSON-only ChartSpec, validates it with Pydantic, and emits it as a
`{"node": "visualize", "content": <spec_json>}` SSE event. The frontend
parses it and renders via ECharts (with a PNG-export button).

Hard rules (per SPEC.md §2 ChartSpec contract):
- `data` is plain JSON — arrays of objects, no functions, no DOM nodes.
- `confidence="extracted"` requires `citation`.
- `confidence="estimated"` requires `note`. No silent fabrications.
- The frontend rejects unknown `type` values; the server validates with
  this Pydantic schema before streaming.

Why a node (not a tool)? A LangChain `@tool` has a fixed signature the
LLM can't expand — there's no way for the LLM to "invent" the
arbitrary ChartSpec shape from a static arg list. A dedicated node that
calls the LLM with a JSON-only system prompt is the clean way to get
arbitrary structured output.
"""

import json
import re
from pathlib import Path
from typing import Literal, Optional

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, model_validator

# Local prompt loader — kept in viz.py to avoid a circular import with
# graph.py (which imports `make_visualize` from this module).
_PROMPT_DIR = Path(__file__).parent.parent / "prompts"


def _load_viz_prompt() -> str:
    return (_PROMPT_DIR / "visualize.txt").read_text(encoding="utf-8")


VISUALIZE_PROMPT = _load_viz_prompt()


ChartType = Literal["bar", "line", "pie", "scatter", "table"]


class Citation(BaseModel):
    """Source citation linking a chart to its underlying document."""

    filename: str = Field(description="The filename (with extension) the data was lifted from.")
    label: str = Field(description="A label or heading from the source (e.g. 'Table 3 — P0420 frequency').")


class ChartDataPoint(BaseModel):
    """One row in the chart's `data` array."""

    x: str = Field(description="Categorical x-axis value (or numeric, serialized as string for scatter).")
    y: float = Field(description="Numeric y-axis value.")
    label: Optional[str] = Field(default=None, description="Optional display label for the point (e.g. legend override).")


class ChartSpec(BaseModel):
    """The ONLY contract the frontend trusts.

    All fields are Pydantic-validated server-side before SSE emit.
    """

    type: ChartType = Field(description="Chart type — frontend must reject unknown values.")
    title: str = Field(description="Human-readable chart title.")
    x_label: Optional[str] = Field(default=None, description="X-axis label.")
    y_label: Optional[str] = Field(description="Y-axis label.")
    data: list[ChartDataPoint] = Field(description="Plain JSON array of {x, y, label?} points.", min_length=1)
    citation: Optional[Citation] = Field(default=None, description="Source citation. Required when confidence='extracted'.")
    confidence: Literal["extracted", "estimated"] = Field(
        description="'extracted' = lifted directly from a source; 'estimated' = computed/inferred (note required)."
    )
    note: Optional[str] = Field(default=None, description="Explanation when confidence='estimated', or when the chart deviates from the user's request.")

    @model_validator(mode="after")
    def _check_confidence_linkage(self):
        if self.confidence == "extracted" and self.citation is None:
            raise ValueError("confidence='extracted' requires a citation.")
        if self.confidence == "estimated" and not self.note:
            raise ValueError("confidence='estimated' requires a note explaining the basis.")
        return self


def validate_chart_spec(spec_dict: dict) -> ChartSpec:
    """Validate a raw dict against the schema.

    Raises ``pydantic.ValidationError`` on any rule violation. The error
    message is intentionally specific so the agent can self-correct
    (e.g. "confidence='extracted' requires a citation").
    """
    return ChartSpec.model_validate(spec_dict)


# ── JSON extraction ────────────────────────────────────────────────

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_FIRST_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_object(text: str) -> dict | None:
    """Best-effort: pull the first JSON object out of an LLM response.

    Tries, in order: fenced ```json``` block, then the first balanced
    {...} region. Returns None if neither matches.
    """
    m = _JSON_FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = _FIRST_OBJECT_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ── Node factory ───────────────────────────────────────────────────

def make_visualize(response_model):
    """Build the visualize node.

    Reads the most recent `generate_answer` message + retrieved context
    from state, calls the LLM with the JSON-only visualize prompt,
    validates the result against ChartSpec, and emits a HumanMessage
    whose `.content` is the spec JSON (so the existing SSE emit at
    server.py:495-500 forwards it as `{"node": "visualize", "content":
    <spec_json>}`).

    On validation failure, emits a stub spec with confidence="estimated"
    and a `note` explaining the failure — so the user always sees
    *something* (and so the chart-emission test in eval.py can verify
    the path was reached).
    """

    def visualize(state: "RhaState"):
        answer = ""
        context = ""
        for msg in reversed(state["messages"]):
            content = getattr(msg, "content", "")
            if not answer and msg.type == "ai":
                answer = content
            if getattr(msg, "type", "") == "tool" and not context:
                context = content

        prompt = VISUALIZE_PROMPT.format(
            answer=answer or state.get("question", ""),
            context=context or "(no retrieved context available)",
            question=state.get("question", ""),
        )
        raw = response_model.invoke([{"role": "user", "content": prompt}]).content

        spec_dict = _extract_json_object(raw) if isinstance(raw, str) else None
        if spec_dict is None:
            # LLM didn't return parseable JSON — emit a stub explaining why.
            spec = ChartSpec(
                type="table",
                title="Chart unavailable",
                y_label="",
                data=[ChartDataPoint(x="error", y=0, label="LLM did not return a JSON object")],
                confidence="estimated",
                note=f"Raw LLM response (truncated): {(raw or '')[:300]}",
            )
        else:
            try:
                spec = validate_chart_spec(spec_dict)
            except Exception as e:
                # Validation failed — surface the error in `note`.
                spec = ChartSpec(
                    type="table",
                    title="Chart validation failed",
                    y_label="",
                    data=[ChartDataPoint(x="error", y=0, label="see note")],
                    confidence="estimated",
                    note=f"Validation error: {e}. Raw spec: {json.dumps(spec_dict)[:300]}",
                )

        return {"messages": [HumanMessage(content=spec.model_dump_json())]}

    return visualize