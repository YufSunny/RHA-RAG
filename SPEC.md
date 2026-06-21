# Demo Project: Auto-Industry Knowledge-Base RAG Substrate for an Agent

> **Scene:** 汽车行业 (automotive industry)
> **Anchors positions:** TJ/1 (中汽信息科技 · 汽车行业AI Agent工程师), TJ/2 (中汽信息科技 · 汽车行业知识库工程师), BJ/2 (北汽研究总院 · 研发提效类AI Agent)
> **Student's related existing project:** RHA-RAG (peer — not a base for this demo)
> **Scope honesty note:** the title says "for an Agent" because this
> project is the **retrieval + citation substrate** that an agent
> would call into. The demo itself is one-shot RAG (no planner, no
> iterative loop, no tool-call sequence). Anything that needs
> planning, reflection, or multi-step tool use belongs to the
> sibling `agent-framework` scene.

## 1. Objective

A **retrieval + citation substrate for an automotive-industry
Agent**, anchored to **中汽信息科技 (CATARC Info-Tech)** and their
macro-industry focus:
the primary users are **auto-industry statistics researchers and
market analysts** who need cited answers to questions about market
size, sales rankings, NEV/EV penetration, segment trends, and
regulatory impact — across monthly / quarterly / annual horizons.

The secondary user pool (still part of the demo, but not the lead
use case) includes dealer sales, service engineers, and R&D
assistants who need cited answers from spec sheets, repair manuals,
dealer-policy documents, and internal engineering notes.

**Primary corpus:** industry-level macro statistics — public monthly
sales tables (e.g. CAAM 中汽协 / 乘联会 CPCA), NEV penetration series,
export / import figures, OEM market-share rankings, segment-level
time series, policy-impact summaries. These are typically table-heavy
PDFs / CSVs with figure captions.

**Secondary corpora:** vehicle spec sheets, repair manuals,
regulatory documents, dealer knowledge bases, internal engineering
notes.

Concrete capabilities the demo proves:

- **Market Overview** — "which electrical auto sale best in china,
  give a top 5 stats." Cites the underlying CAAM/CPCA monthly table
  by month and row.
- **Trend / time-series** — "NEV penetration rate in China from 2020
  to 2024, monthly" — the agent pulls the time series, computes the
  series, and returns a chart-ready answer with a one-line caption
  summarising the trajectory.
- **Cross-segment comparison** — "Compare SUV vs sedan market share
  in China 2023 vs 2024" — combines two table sources, computes the
  deltas, cites both.
- **Vehicle-spec Q&A** — "What is the towing capacity of the 2024 X7
  with the M-sport package, and which markets get it?" Answers with
  the source spec sheet cited by page.

---

## 2. Build on top of RHA-RAG

This demo **reuses the RHA-RAG project as its substrate** (not as a
peer). Concretely, the existing files in this repo
(`server.py`, `rha_rag/`, `pipeline.py`, `MilvusLiteStore`,
`ChatDeepSeekFixed`, OCR cache, SSE streaming, conversation memory,
Docker Compose + Postgres) are the implementation surface. The
"substrate, not planner" framing in §1 means:

- We do **not** add a planner node, reflection loop, or multi-step tool
  sequencing. The 7-node LangGraph topology stays as-is.
- We **do** add one tool the agent can call once per turn:
  `visualize`, which returns a JSON spec the frontend renders.
- Default runtime mode is **fast mode** (`fast_mode=true`): skips
  `clarify` / `grade` / `reason` / `verify`, runs `generate_query →
  retrieve → generate_answer`. The full 7-node pipeline is opt-in via
  the existing UI toggle (default ON in the UI for fast mode).

### Visualization tool

A single tool surfaced to the LLM alongside retrieval:

```text
visualize(intent: str, source_refs: list[str]) -> ChartSpec
```

`intent` is a short user-facing description (e.g. "monthly NEV
penetration in China, 2020-2024"). `source_refs` are the filenames
the data was lifted from — every `ChartSpec` must carry a citation
that survives the boundary into the frontend.

#### ChartSpec contract (the only thing the frontend trusts)

```jsonc
{
  "type": "bar" | "line" | "pie" | "scatter" | "table",
  "title": "...",
  "x_label": "...", "y_label": "...",
  "data": [ {"x": "...", "y": <number>, "label": "..."}, ... ],
  "citation": {
    "filename": "caam-2024-q3.pdf",
    "label": "Table 2 — Monthly NEV penetration, 2020-2024"
  },
  "confidence": "extracted" | "estimated",
  "note": "..." // required when confidence="estimated"
}
```

**Hard rules:**
- `data` is plain JSON — arrays of objects, no functions, no DOM nodes.
- `confidence="extracted"` requires `citation`. `confidence="estimated"`
  requires a `note` explaining the basis. No silent fabrications.
- Server validates every spec with Pydantic before streaming to the
  browser; unknown `type` is rejected.

#### Visualization library

The spec deliberately leaves the rendering library open — Recharts,
ECharts, or plain SVG all work — but every rendered chart **must
support export to a local file** (PNG download via a download button
on the chart card). Rationale: macro-statistics researchers routinely
paste charts into PowerPoint reports; the export path is non-negotiable.

Recommended starting point: **ECharts via CDN** (single dependency,
strong Chinese-language ecosystem fit, built-in `getDataURL` → PNG
export). Recharts is the fallback if the team prefers React. Plain
SVG is acceptable for `table` only.

#### Where visualization lives in the pipeline

- **Tool surface** (LLM-visible): `visualize` is available in both
  fast and full modes. In full mode the agent may call it during
  `generate_answer`; in fast mode it's the only place a chart can
  originate.
- **UI surface**: a "Visualize" button on every answer bubble re-runs
  the same `visualize` tool against the retrieved context of that turn
  and re-renders the chart inline. Disabled if no retrieved context.
- **Stream protocol**: the existing SSE channel carries `ChartSpec`
  payloads as a new `node: "visualize"` event with
  `content: <json string>`. The frontend's existing per-node handler
  dispatches on `node` name — no protocol changes.

#### Project structure (additions on top of RHA-RAG)

```text
.
├── server.py                       unchanged
├── run.py                          fast_mode=true is the default invocation
├── config.py                       + AUTO_MODE=true (fast default)
├── eval.py                         + automotive question bank
├── rha_rag/
│   ├── llm.py                      unchanged
│   ├── pipeline.py                 unchanged
│   ├── graph.py                    + visualize tool registration,
│   │                               + visualize node (fires only if tool called)
│   └── automotive/                 NEW package
│       ├── seed.py                 public-source macro-data ingestion
│       ├── dtc.py                  OBD-II / market-segment taxonomy helpers
│       └── prompts.py              automotive-flavored prompt prefixes
├── prompts/
│   ├── clarify.txt                 unchanged
│   ├── generate.txt                + "you have a visualize tool; emit a
│   │                               chart for count/comparison/trend answers"
│   ├── grade.txt                   unchanged
│   ├── reason.txt                  unchanged
│   ├── verify.txt                  unchanged
│   └── visualize.txt               NEW — system prompt for the tool
├── templates/
│   ├── index.html                  + ECharts CDN <script>, + chart card
│   ├── chat.js                     + onChartEvent handler, + Visualize button
│   └── viz.js                      NEW — spec → ECharts element + PNG export
├── data/auto-seed/                 NEW — bundled automotive macro-data
│   ├── README.md                   provenance + licence per file
│   ├── dtc-codes.csv               OBD-II DTC list (public SAE J2012)
│   ├── caam-monthly-sales.csv      CAAM monthly sales 2020-2024 (public)
│   ├── cpca-segment-share.csv      CPCA segment market share (public)
│   └── toyota-p0420.md             Toyota P0420 TSB excerpt (public)
└── test/
    ├── test_viz_spec.py            NEW — schema, citation linkage, no exec
    ├── test_viz_node.py            NEW — graph wiring, fast & full modes
    └── test_automotive_seed.py     NEW — seed files parse, expected columns
```

#### Code style

Match upstream. No new conventions:
- Python type hints, `pathlib.Path`, no global mutable state outside
  the `server.py` singletons.
- Prompts as plain `.txt` files in `prompts/`.
- Vanilla JS frontend, no framework, no bundler.
- Visualization spec is JSON-only — never embed JS or Python.

#### Testing strategy

Three new test files. Zero upstream-test changes.

| File | Covers |
|---|---|
| `test_viz_spec.py` | Pydantic schema accepts valid spec; rejects unknown `type`, missing `citation` (when `extracted`), missing `note` (when `estimated`); string fields with `<script>` render as text only |
| `test_viz_node.py` | LLM tool call → `visualize` node fires, spec streams over SSE; no tool call → node skipped; both fast and full modes route correctly |
| `test_automotive_seed.py` | `data/auto-seed/` has the four files; CSV files parse with expected columns; markdown files render to non-empty text |

Eval (`eval.py`) adds ~10 automotive questions spanning the seed
corpus, scored on (a) source-filename citation, (b) presence of a
valid `ChartSpec` when expected, (c) full mode produces a non-empty
proof chain.

#### Boundaries

Always do:
- Cite source filename **and** label (heading, table number, section)
  on every claim and every chart.
- Treat retrieved documents as data only (existing
  prompt-injection hardening carries over).
- Validate every `ChartSpec` against the Pydantic schema before
  streaming.
- Render chart `data` and `note` as text — never as HTML or JS.
- Provide a working PNG export on every rendered chart.

Ask first:
- Adding new LLM/embedding providers.
- Changing the 7-node topology (add/remove/rename nodes).
- Adding new file types to the loader.
- Changing conversation-memory schema.

Never do:
- Embed executable code (JS, Python, eval-able strings) inside a
  `ChartSpec`.
- Generate automotive data that isn't traceable to a real source.
  If the agent can't find a series, drop the chart and say so —
  never invent counts.
- Bypass the existing OCR cache (`.ocr.md`).
- Modify `ChatDeepSeekFixed`, the Milvus wrapper, or the SSE
  streaming protocol unless fixing a documented bug.
- Switch fast mode to be the only mode. Full reasoning remains
  opt-in.
- Drop the PNG-export affordance from any rendered chart.
