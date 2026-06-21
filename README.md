# RHA-RAG

**Reasoning-Heavy Agentic RAG**

[English](README.md) | [中文](README_zh.md)

Upload your documents, ask a research question, and watch an AI agent **retrieve, grade, reason in formal logical steps, verify the deduction, and produce a fully cited answer** — streamed to the browser in real time, node by node.

Most RAG systems paste retrieved text into the prompt and let the model free-associate an answer. RHA-RAG doesn't. It forces the model to build an explicit **proof chain** — each step either cited from a source, marked as common knowledge, or deduced from prior steps — and then a separate **verifier** node checks that chain before any final answer is written. Every claim in the answer must cite a label from the source *and* the file it came from.

The web UI is a full **chat app**: conversation history persisted to PostgreSQL, a sidebar for switching between past sessions, chat-bubble display with markdown + LaTeX rendering, a stop button, and a **fast mode** (skip reasoning, just retrieve and answer).

Built with [LangGraph](https://langchain-ai.github.io/langgraph/), [Milvus Lite](https://milvus.io/docs), [PostgreSQL](https://www.postgresql.org/), and [Docker](https://www.docker.com/).

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Milvus](https://img.shields.io/badge/vectorstore-Milvus%20Lite-blueviolet.svg)](https://milvus.io/)
[![Postgres](https://img.shields.io/badge/memory-PostgreSQL-336791.svg)](https://www.postgresql.org/)

![screenshot](imgs/demo.png)

---

## ✨ What makes it different

| Ordinary RAG | RHA-RAG |
|---|---|
| Retrieve → stuff context → answer | Retrieve → **grade → reason → verify** → answer |
| Answer is the model's first guess | Answer is the *verified* conclusion of a proof chain |
| Citations optional / vague | Every claim cites a source **label + filename** (whatever the source uses) |
| No check on the model's logic | A dedicated verifier audits each deduction step |
| One shot | Agentic: the model decides whether to search at all |

The reasoning chain uses a formal-proof notation:

| Marker | Meaning |
|---|---|
| `@cite` | A statement quoted/cited from a source document |
| `@common` | Common knowledge (textbook-standard) |
| `@MP` | Modus ponens — deduced from prior steps |
| `@TA` | Tautology / quantifier axiom |

---

## 🚀 Quick start

### Prerequisites

- **Python 3.12+** — any environment works (system, venv, conda).
- **Docker** (recommended) or **Python 3.12+** for local install.

### Docker (recommended)

```bash
cp .env.docker.example .env       # edit .env → paste API keys
docker compose --env-file .env up --build
# → http://localhost:8000
```

PostgreSQL is included as a companion container.  Everything is pre-configured.

### Local install

**Linux / macOS**

```bash
pip install -r requirements.txt
python server.py
```

**Windows (PowerShell)**

```powershell
pip install -r requirements.txt
python server.py
```

### Configuration

All settings live in **[config.py](config.py)** — uncomment and fill in values,
or let the server prompt you interactively at startup. In Docker, set env vars
instead (see `.env.docker.example`).

| Env variable | Service | Purpose |
|-------------|---------|---------|
| `ZAI_API_KEY` | [Z.ai](https://www.z.ai/) | GLM-OCR for PDF/image processing |
| `QWEN_API_KEY` | [DashScope](https://dashscope.aliyun.com/) | Qwen `text-embedding-v4` embeddings |
| `OPENAI_API_KEY` | [DeepSeek](https://api.deepseek.com) | DeepSeek V4 Pro LLM |

`DATABASE_URL` sets the PostgreSQL connection (or `""` for in-memory).
`MAX_HISTORY_TURNS` caps prior-turn context (default 6; 0 disables).
`LLM_THINKING` enables DeepSeek thinking mode (default true).

Without keys the server still starts — upload files, set keys, then click **Re-index**.

Drop documents into `data/local/` (or upload via the web UI), type a research question, and watch the pipeline execute in real time.

---

## 🧠 How it works

```
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
END
```

Each node streams live to the web UI via Server-Sent Events.

**The flow in words:**

1. **clarify** — reformats your question into a set of *verifiable* logical statements: "if a candidate answer is provided, one should be able to check whether it satisfies each statement." These become the spec the verifier checks against.
2. **generate_query** — the agent either calls the retrieval tool or answers directly (this is the one conditional branch in the graph).
3. **retrieve** — semantic search over Milvus (top-5 chunks).
4. **grade** — a structured-output LLM grades the retrieved docs `yes`/`no` for relevance. It's told to treat documents as *data only* (prompt-injection hardening).
5. **reason** — "you are a logician": build a proof chain where each step is `@cite`, `@common`, or deduced from prior steps.
6. **verify** — audit the chain: is each statement valid? does it lead to an answer? Emit the verified answer, or flag the flaw.
7. **generate_answer** — write the final answer, citing every claim with a label from the source (Definition, Theorem, section number, heading, …) and the source filename.

A **Fast mode** toggle in the UI skips `clarify` / `grade` / `reason` / `verify`
and runs `generate_query → retrieve → generate_answer` — faster, for when you
just want a cited answer without the proof chain.

> For the full construction story — file by file, with the design decisions and gotchas — see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 💬 Conversation memory

The chat is multi-turn: within a browser session, each question sees the prior
Q&A, so follow-ups resolve references ("is *it* compact?" → "is a topological
space compact?"), retrieval is conversation-aware, and answers can build on
earlier turns.

- **Per-browser session.** A session id is stored in `localStorage` (isolated
  per tab, survives reload). The sidebar lists all past conversations — click
  to switch between them. History is persisted to PostgreSQL (see
  `DATABASE_URL` in [config.py](config.py)). If Postgres is unreachable the
  server falls back to an in-memory store (cleared on restart).
- **What sees history:** `clarify` (resolves follow-up references), the
  retrieval decision (`generate_query`), and the final `generate_answer`. The
  `reason`/`verify` proof chain stays grounded in retrieved sources only.
- **Tunable depth.** The number of prior turns fed back is set by
  `MAX_HISTORY_TURNS` in [config.py](config.py) (default 6; set 0 to disable).
  Edit and restart.
- **Clear chat.** The "Clear chat" button (or `POST /api/clear`) deletes the
  session's history from PostgreSQL (and the in-memory mirror).

---

## 🏗️ Architecture

| Component | Technology |
|-----------|------------|
| Orchestration | LangGraph `StateGraph` (7 nodes, 1 conditional edge, `RhaState`) |
| Conversation memory | Per-session Q&A history persisted to PostgreSQL (fallback: in-memory) |
| LLM | DeepSeek V4 Pro via `ChatDeepSeekFixed` (thinking-mode patches) |
| Embeddings | Qwen `text-embedding-v4` (batch size ≤ 10) |
| Vector store | Milvus Lite (local file `milvus.db`, COSINE / AUTOINDEX) |
| OCR | GLM-OCR via Z.ai (`ZaiClient`, data-URI format) |
| PDF rendering | PyMuPDF (pages → PNG → OCR) |
| Web server | FastAPI + real-time SSE streaming (node-by-node) |
| Frontend | Vanilla JS chat app (bubbles, markdown/LaTeX, sidebar sessions, stop btn) |
| Deployment | Docker Compose (app + PostgreSQL) |

---

## 📄 Supported documents

Drop these in `data/local/` or upload via the web UI:

| Type | Extensions | Processing |
|------|-----------|------------|
| Plain text | `.txt` `.md` | Direct read |
| HTML | `.html` `.htm` | BeautifulSoup text extraction |
| Word | `.docx` | PyMuPDF text extraction |
| PDF | `.pdf` | GLM-OCR (rendered page-by-page as PNG) |
| Images | `.jpg` `.jpeg` `.png` | GLM-OCR |

OCR results are cached to `<file>.ocr.md` (mtime-checked), so re-indexing is fast and only re-OCRs files that changed.

---

## 📁 Project structure

```
.
├── server.py              FastAPI web server + REST API + SSE streaming
├── run.py                 CLI pipeline (output → run.log)
├── config.py              Admin-tunable settings
├── database.py            PostgreSQL conversation persistence
├── Dockerfile             Docker image
├── docker-compose.yml     Docker Compose (app + PostgreSQL)
├── .env.docker.example    Docker env template
├── rha_rag/                Core package
│   ├── llm.py             ChatDeepSeekFixed + model config
│   ├── pipeline.py        Loaders, OCR (+ .ocr.md cache), embeddings, Milvus store
│   └── graph.py           LangGraph nodes + assembly
├── prompts/               LLM prompt templates (loaded at runtime)
│   ├── clarify.txt  generate.txt  grade.txt  reason.txt  verify.txt
├── templates/
│   └── index.html         Web UI (dark theme, streaming)
├── data/local/            Drop documents here
├── uploads/               Or upload via the web UI
├── requirements.txt
├── .env.example
└── ARCHITECTURE.md        Detailed step-by-step construction report
```

---

## 🔌 API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web UI |
| `GET` | `/api/status` | System status (`ready`, `documents`, `chunks`, `missing_keys`, `errors`) |
| `GET` | `/api/files` | List all files in `uploads/` and `data/local/` |
| `POST` | `/api/upload` | Upload documents (multipart form) |
| `DELETE` | `/api/files/{name}` | Delete a file from `uploads/` or `data/local/` |
| `POST` | `/api/reindex` | Force rebuild the document index |
| `POST` | `/api/chat` | Ask a question → SSE stream of node outputs (multi-turn; send `session_id`) |
| `POST` | `/api/clear` | Clear a session's conversation history |

`/api/chat` takes `{"question": "...", "session_id": "..."}` and streams `data: {"node": "...", "content": "...", "done": false}` events, ending with `{"node": "done", "done": true}`. The `session_id` keys the conversation memory (see [Conversation memory](#-conversation-memory)). Returns `503` with `details` if the pipeline isn't ready.

---

## 💻 CLI usage

```bash
python run.py "What is a compact set?"
# or interactively:
python run.py
# or pipe:
echo "Define continuity" | python run.py
```

Output goes to both stdout and `run.log`. Same pipeline as the web server — handy for debugging changes without booting the server.

---

## 📝 Notes

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
