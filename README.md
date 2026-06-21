# visualized RAG for automotive industry: demo

**An RAG system for the automotive industry, built on top of [RHA-RAG](https://github.com/...)'s proof-chain engine, with chart visualization as a first-class tool-calling capability.**

[English](README.md) | [中文](README_zh.md)

Ask *"How many BEVs vs PHEVs did Automaker A sell in 2024?"* and RAG for automotive industry demo **retrieves** the relevant row from your automotive corpus, **cites** the source file and label, emits a strict-JSON `ChartSpec` for the numbers, and renders a full-width **ECharts** bar chart with a one-click **⬇ PNG** download — streamed node by node to the browser via SSE. Switch to **Full** mode and the same question goes through an explicit proof chain (clarify → grade → reason → verify → answer → visualize) with each step auditable.

The bundled seed corpus is **Automaker A** — annual/quarterly/model-level sales, global market share, and the City-EV launch — so the system answers real automotive questions on first boot with zero uploads. Drop your own CAAM / CPCA / OEM / TSB / service-manual files alongside and the same pipeline answers them.

Built on [LangGraph](https://langchain-ai.github.io/langgraph/) (orchestration), [Milvus Lite](https://milvus.io/docs) (vector store), [PostgreSQL](https://www.postgresql.org/) (conversation memory), [ECharts](https://echarts.apache.org/) (chart rendering), and [Docker](https://www.docker.com/).

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![Domain](https://img.shields.io/badge/domain-Automotive%20%2F%20NEV-0066cc.svg)](#-what-this-system-does)
[![Charts](https://img.shields.io/badge/charts-ECharts%205-aa2233.svg)](https://echarts.apache.org/)
[![Orchestration](https://img.shields.io/badge/orchestration-LangGraph-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Vector store](https://img.shields.io/badge/vectorstore-Milvus%20Lite-blueviolet.svg)](https://milvus.io/)
[![Memory](https://img.shields.io/badge/memory-PostgreSQL-336791.svg)](https://www.postgresql.org/)

![screenshot](imgs/demo.png)

---

## ✨ What this system does

RAG for automotive industry demo is a **domain-tuned RAG + visualization tool-calling** system for the automotive industry. It is not a generic chatbot. The two non-negotiable capabilities:

### 1. Cited answers, not free-association

Every claim in the final answer cites a **label** from the source (e.g. `Table 3 — P0420 frequency by model year`) **and** the source **filename** (e.g. `toyota-p0420.md`). A retrieved document is treated as *data only* — never as instructions — so prompt-injected text inside a PDF cannot redirect the pipeline.

### 2. Chart as a tool-calling capability

When the answer contains numeric data (sales, frequencies, market share, monthly trends), a second LLM call emits a **Pydantic-validated `ChartSpec`**:

```text
{type, title, x_label, y_label, data[], citation{filename,label}, confidence, note}
```

The frontend renders it as a full-width ECharts card with axes, tooltips, data-zoom (for time series), and a **⬇ PNG** export button. The chart and the prose answer are linked by the same `citation`. A chart without a source citation is rejected at the Pydantic layer.

The chart works in **both** modes:

- **Fast mode (default)** — `generate_query → retrieve → generate_answer → visualize`. Low latency for "just answer + chart" workflows.
- **Full mode** — `clarify → generate_query → retrieve → grade → reason → verify → generate_answer → visualize`. Adds the proof chain when the question is novel or high-stakes.

---

## 🚀 Quick start

### Prerequisites

- **Docker** (recommended) — includes PostgreSQL.
- **Python 3.12+** (alternative) — for local install.

### Docker (recommended)

```bash
cp .env.docker.example .env          # paste API keys
docker compose --env-file .env up --build
# → http://localhost:7500  (PORT env var, default 7500)
```

PostgreSQL is included as a companion container. No external services required.

### Local install

#### Linux / macOS

```bash
pip install -r requirements.txt
python server.py
```

#### Windows (PowerShell)

```powershell
pip install -r requirements.txt
python server.py
```

### Configuration

All settings live in **[config.py](config.py)**. In Docker, set env vars in `.env`.

| Env variable | Service | Purpose |
| --- | --- | --- |
| `ZAI_API_KEY` | [Z.ai](https://www.z.ai/) | GLM-OCR for PDF / image processing |
| `QWEN_API_KEY` | [DashScope](https://dashscope.aliyun.com/) | Qwen `text-embedding-v4` embeddings |
| `OPENAI_API_KEY` | [DeepSeek](https://api.deepseek.com) | DeepSeek V4 Pro LLM |
| `PORT` | — | Host:container port (default `7500`) |
| `DEFAULT_FAST_MODE` | — | `true` / `false` — default mode for `/api/chat` (default `true`) |

`DATABASE_URL` sets the PostgreSQL connection (or `""` for in-memory). `MAX_HISTORY_TURNS` caps prior-turn context (default 6; 0 disables). `LLM_THINKING` enables DeepSeek thinking mode (default `true`).

Without API keys the server still starts — upload files, set keys, then click **Re-index**.

---

## 🧠 How it works

### Fast mode (default)

```text
generate_query → retrieve → generate_answer → visualize → END
```

### Full mode (proof chain)

```text
User Question
    │
    ▼
clarify          Translate natural language into goal-driven logical statements
    │
    ▼
generate_query   LLM decides: search the knowledge base, or answer directly?
    │
    ├──(no tool call)── END
    │
    ▼
retrieve         Semantic search over the local Milvus vector store
    │
    ▼
grade            Assess document relevance with structured LLM output (yes/no)
    │
    ▼
reason           Build a logical proof chain (@cite / @common / @MP / @TA)
    │
    ▼
verify           Validate each deduction step against inference rules
    │
    ▼
generate_answer  Produce the final answer with explicit source citations
    │
    ▼
visualize        Tool call: emit a ChartSpec (Pydantic) for any numeric answer
    │
    ▼
END
```

Each node streams live to the web UI via Server-Sent Events. Click **Full** in the top bar to switch modes; click **Fast** to return.

**Node responsibilities in an automotive context:**

1. **clarify** — reformats the question into verifiable statements. *"Is the claim that Automaker A's 2024 BEV sales were 1,764,992 units?"*
2. **generate_query** — decides whether to call `retrieve` or answer directly.
3. **retrieve** — semantic search over Milvus (top-5 chunks) — CSV rows, markdown sections, TSB text, all indexed as plain text.
4. **grade** — a structured-output LLM grades each retrieved chunk `yes`/`no` for relevance. Documents are treated as *data only* (prompt-injection hardening — a PDF pretending to be instructions is rejected).
5. **reason** — "you are a logician": build a proof chain where each step is `@cite`, `@common`, or deduced from prior steps.
6. **verify** — audit the chain. Flag contradictions, missing citations, or unsound deductions before the answer is written.
7. **generate_answer** — write the final answer, citing every claim with a label from the source (Table number, section heading, row range) and the source filename.
8. **visualize** — **tool call.** A second LLM call inspects the answer + retrieved context and emits a strict-JSON `ChartSpec`. Pydantic-validated server-side; rejected specs never reach the frontend. Rendered client-side with ECharts.

The reasoning chain uses a formal-proof notation:

| Marker | Meaning |
| --- | --- |
| `@cite` | A statement quoted/cited from a source document |
| `@common` | Common knowledge (textbook-standard) |
| `@MP` | Modus ponens — deduced from prior steps |
| `@TA` | Tautology / quantifier axiom |

> For the full construction story — file by file, with the design decisions and gotchas — see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 💬 Conversation memory

The chat is multi-turn: within a browser session, each question sees the prior
Q&A, so follow-ups resolve references (*"is it compact?"* → *"is a topological
space compact?"*; *"which of those had the biggest jump?"* → references the
prior model's answer), retrieval is conversation-aware, and answers can build
on earlier turns.

- **Per-browser session.** A session id is stored in `localStorage` (isolated
  per tab, survives reload). The sidebar lists all past conversations — click
  to switch. History is persisted to PostgreSQL. If Postgres is unreachable
  the server falls back to an in-memory store (cleared on restart).
- **What sees history:** `clarify` (resolves follow-up references), the
  retrieval decision (`generate_query`), and the final `generate_answer`. The
  `reason` / `verify` proof chain stays grounded in retrieved sources only.
- **Tunable depth.** `MAX_HISTORY_TURNS` in [config.py](config.py) (default 6;
  set 0 to disable).
- **Clear chat.** The "Clear chat" button (or `POST /api/clear`) deletes the
  session's history from PostgreSQL.

---

## 🏗️ Architecture

| Component | Technology |
| --- | --- |
| Domain | **Automotive industry** — OEMs, NEV market data, TSBs, service manuals, DTCs |
| Visualization | **ECharts 5** via CDN; `ChartSpec` Pydantic model; PNG export via `getDataURL` |
| Orchestration | LangGraph `StateGraph` (8 nodes, 1 conditional edge, `RhaState`) |
| Proof chain | Formal `@cite` / `@common` / `@MP` / `@TA` step notation, verifier node |
| Conversation memory | Per-session Q&A history persisted to PostgreSQL (fallback: in-memory) |
| LLM | DeepSeek V4 Pro via `ChatDeepSeekFixed` (thinking-mode patches) |
| Embeddings | Qwen `text-embedding-v4` (batch size ≤ 10) |
| Vector store | Milvus Lite (local file `milvus.db`, COSINE / AUTOINDEX) |
| OCR | GLM-OCR via Z.ai (`ZaiClient`, data-URI format) |
| PDF rendering | PyMuPDF (pages → PNG → OCR) |
| Web server | FastAPI + real-time SSE streaming (node-by-node) |
| Frontend | Vanilla JS chat app — bubbles, markdown / LaTeX, sidebar sessions, ECharts card, PNG export, fast/full toggle |
| Deployment | Docker Compose (app + PostgreSQL), default port 7500 |

---

## 📄 Supported documents

Drop these in `data/local/` or upload via the web UI:

| Type | Extensions | Processing |
| --- | --- | --- |
| Plain text | `.txt` `.md` | Direct read |
| **CSV** (sales tables, DTC lists) | `.csv` | Direct read (header + rows joined as text) |
| HTML | `.html` `.htm` | BeautifulSoup text extraction |
| Word | `.docx` | PyMuPDF text extraction |
| PDF (TSBs, service manuals) | `.pdf` | GLM-OCR (rendered page-by-page as PNG) |
| Images (dashboards, photos) | `.jpg` `.jpeg` `.png` | GLM-OCR |

OCR results are cached to `<file>.ocr.md` (mtime-checked), so re-indexing is fast and only re-OCRs files that changed.

### Bundled seed corpus

`data/auto-seed/` ships with five public-source **Automaker A** documents so the demo works on first boot with zero uploads:

- `automaker-annual.csv` — annual production, sales, BEV/PHEV split, China vs overseas 2019–2025
- `automaker-quarterly.csv` — quarterly sales 2023-Q2 → 2025-Q4
- `automaker-models-2025.csv` — top 13 Automaker A model families 2025 with YoY change
- `automaker-market-share.md` — global plug-in EV share 2025, China NEV share 2024 (CPCA)
- `automaker-city-ev.md` — City-EV product profile (Automaker A Global press release, 30 June 2025)

All data is from public press releases, CPCA / CAAM monthly reports, Statista compilations, and CnEVPost reporting. No proprietary or restricted material. See [data/auto-seed/README.md](data/auto-seed/README.md) for sources and licensing.

Drop your own CAAM monthly sales, CPCA segment share, OEM TSBs, or service manuals into `data/local/` to extend the system to your fleet / brand.

---

## 📁 Project structure

```text
.
├── server.py              FastAPI web server + REST API + SSE streaming
├── run.py                 CLI pipeline (output → run.log)
├── config.py              Admin-tunable settings
├── database.py            PostgreSQL conversation persistence
├── Dockerfile             Docker image
├── docker-compose.yml     Docker Compose (app + PostgreSQL), default PORT=7500
├── .env                   Local env (API keys + PORT)
├── rha_rag/                Core package
│   ├── llm.py             ChatDeepSeekFixed + model config
│   ├── pipeline.py        Loaders, OCR (+ .ocr.md cache), embeddings, Milvus store
│   ├── graph.py           LangGraph nodes + assembly
│   └── viz.py             ChartSpec Pydantic model + visualize (tool-call) node
├── prompts/               LLM prompt templates (loaded at runtime)
│   ├── clarify.txt  generate.txt  grade.txt  reason.txt  verify.txt  visualize.txt
├── templates/
│   └── index.html         Web UI (dark theme, ECharts, streaming)
├── data/
│   ├── local/             Drop your automotive documents here
│   ├── uploads/           Or upload via the web UI
│   └── auto-seed/         Automaker A seed corpus (auto-loaded on boot)
├── tests/                 pytest unit tests (43 tests, no LLM required)
├── test/ground_truth.json Eval question set (5 math + 9 Automaker A)
├── eval.py                LLM-judge evaluation harness
├── requirements.txt
├── .env.example
└── ARCHITECTURE.md        Detailed step-by-step construction report
```

---

## 🔌 API endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Web UI |
| `GET` | `/api/status` | System status (`ready`, `documents`, `chunks`, `missing_keys`, `errors`) |
| `GET` | `/api/files` | List all files in `uploads/` and `data/local/` |
| `POST` | `/api/upload` | Upload documents (multipart form) |
| `DELETE` | `/api/files/{name}` | Delete a file from `uploads/` or `data/local/` |
| `POST` | `/api/reindex` | Force rebuild the document index |
| `POST` | `/api/chat` | Ask a question → SSE stream of node outputs (multi-turn; send `session_id`) |
| `POST` | `/api/clear` | Clear a session's conversation history |

`/api/chat` takes `{"question": "...", "session_id": "...", "fast": true}` and streams `data: {"node": "...", "content": "...", "done": false}` events, ending with `{"node": "done", "done": true}`. The `visualize` event's `content` is a JSON-encoded `ChartSpec` (validated server-side). The `session_id` keys the conversation memory (see [Conversation memory](#-conversation-memory)). Returns `503` with `details` if the pipeline isn't ready.

---

## 💻 CLI usage

```bash
python run.py "How many BEVs did Automaker A sell in 2024?"
# or interactively:
python run.py
# or pipe:
echo "What is the China NEV market share for Automaker A in 2024?" | python run.py
```

Output goes to both stdout and `run.log`. Same pipeline as the web server — handy for debugging changes without booting the server.

---

## 📝 Notes

### ChartSpec transport

Charts flow through the existing SSE channel — no protocol change. The `visualize` node emits a `HumanMessage` whose `.content` is the JSON spec. The frontend parses it, builds an ECharts option (bar / line / pie / scatter / table), and renders a full-width chart card. PNG export uses `chart.getDataURL({ type: 'png', pixelRatio: 2, backgroundColor: '#0d1117' })` — works offline, no server roundtrip.

### DeepSeek V4 patch

`ChatDeepSeekFixed` patches three incompatibilities with DeepSeek V4 thinking mode:

1. **`reasoning_content` preservation** — required across tool-call round-trips; LangChain strips it.
2. **List content serialization** — tool/assistant messages with list content must be stringified.
3. **`tool_choice` demotion** — thinking mode rejects `{"type":"function",...}`; forced to `"auto"`.

Both LLM clients use `max_retries=5` so transient upstream disconnects ("Server disconnected without sending a response") are retried instead of failing the run. See [langchain-ai/langchain#37178](https://github.com/langchain-ai/langchain/issues/37178).

### Why Milvus Lite (and not `langchain-milvus`)?

The project uses a small `MilvusClient` wrapper (`MilvusLiteStore` in `rha_rag/pipeline.py`) over a local `milvus.db` file — no server to run. `langchain-milvus` 0.3.3 is incompatible with `pymilvus` 2.6.x (its ORM `Collection` path can't resolve the connection alias that `MilvusClient` registers). The wrapper talks to `MilvusClient` directly and sidesteps that. Full rationale in [ARCHITECTURE.md §7](ARCHITECTURE.md).

---

## 📜 License

MIT — see [LICENSE](LICENSE).
