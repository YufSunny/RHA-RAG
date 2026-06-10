# Cogent

**Reasoning-Heavy Agentic RAG** — upload your documents, ask research questions, and watch an AI agent retrieve, grade, reason logically, verify deductions, and produce cited answers. Built with [LangGraph](https://langchain-ai.github.io/langgraph/).

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)

<hr>

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Set API keys
export ZAI_API_KEY="..."      # Z.ai — GLM-OCR
export QWEN_API_KEY="..."     # Qwen/DashScope — embeddings
export OPENAI_API_KEY="..."   # DeepSeek — LLM

# 3. Launch
python server.py              # → http://localhost:8000
```

Upload documents (or drop them in `data/local/`), type a research question, and watch the pipeline execute in real time.

## How It Works

```
User Question
    │
    ▼
clarify          Translate natural language into goal-driven logical statements
    │
    ▼
generate_query   LLM decides: search the knowledge base, or answer directly
    │
    ├──(no tool call)── END
    │
    ▼
retrieve         Semantic search over the local vector store
    │
    ▼
grade            Assess document relevance with structured LLM output
    │
    ▼
reason           Build a logical proof chain (@cite / @common / @MP / @TA)
    │
    ▼
verify           Validate each deduction step against inference rules
    │
    ▼
generate_answer  Produce the final answer with explicit source citations
```

Each node streams live to the web UI via Server-Sent Events.

## Architecture

| Component | Technology |
|-----------|-----------|
| Orchestration | LangGraph `StateGraph` (7 nodes, conditional edges) |
| LLM | DeepSeek V4 Pro via `ChatDeepSeekFixed` |
| Embeddings | Qwen `text-embedding-v4` (batch size ≤10) |
| OCR | GLM-OCR via Z.ai (`ZaiClient`, data URI format) |
| Vector Store | LangChain `InMemoryVectorStore` |
| PDF Rendering | PyMuPDF (pages → PNG → OCR) |
| Web Server | FastAPI + SSE streaming |

## Supported Documents

Drop these in `data/local/` or upload via the web UI:

| Type | Extensions | Processing |
|------|-----------|------------|
| Plain text | `.txt` `.md` | Direct read |
| HTML | `.html` `.htm` | BeautifulSoup text extraction |
| Word | `.docx` | PyMuPDF |
| PDF | `.pdf` | GLM-OCR (rendered page-by-page as PNG) |
| Images | `.jpg` `.jpeg` `.png` | GLM-OCR |

## Project Structure

```
.
├── server.py              FastAPI web server + API
├── run.py                 CLI pipeline (output → run.log)
├── cogent/                Core package
│   ├── llm.py             ChatDeepSeekFixed + model config
│   ├── pipeline.py        Document loaders, OCR (+ .ocr.md caching), embeddings, vector store
│   └── graph.py           LangGraph nodes + assembly
├── prompts/               LLM prompt templates (loaded at runtime)
├── templates/
│   └── index.html         Web UI (dark theme, streaming)
├── data/local/            Pre-loaded documents (sample set included)
├── uploads/               User-uploaded documents
├── requirements.txt
├── .env.example
└── LICENSE
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web UI |
| `GET` | `/api/status` | System status (`ready`, `documents`, `chunks`, `errors`) |
| `GET` | `/api/files` | List all files in `uploads/` and `data/local/` |
| `POST` | `/api/upload` | Upload documents (multipart form) |
| `DELETE` | `/api/files/{name}` | Delete an uploaded file |
| `POST` | `/api/reindex` | Force rebuild the document index |
| `POST` | `/api/chat` | Ask a question → SSE stream of node outputs |

## CLI Usage

```bash
python run.py "What is a compact set?"
# or interactively:
python run.py
# or pipe:
echo "Define continuity" | python run.py
```

Output goes to both stdout and `run.log`. Same pipeline as the web server.

## Configuration

Copy `.env.example` to `.env`:

| Variable | Service | Purpose |
|----------|---------|---------|
| `ZAI_API_KEY` | [Z.ai](https://www.z.ai/) | GLM-OCR for PDF/image processing |
| `QWEN_API_KEY` | [DashScope](https://dashscope.aliyun.com/) | Qwen text-embedding-v4 |
| `OPENAI_API_KEY` | [DeepSeek](https://api.deepseek.com) | DeepSeek V4 Pro LLM |

Without keys, the server still starts — upload files, then set keys and click **Re-index**.

## DeepSeek V4 Patch

`ChatDeepSeekFixed` patches three incompatibilities with DeepSeek V4 thinking mode:

1. **`reasoning_content` preservation** — required across tool-call round-trips; LangChain strips it
2. **List content serialization** — tool/assistant messages with list content must be stringified
3. **`tool_choice` demotion** — thinking mode rejects `{"type":"function",...}`; forced to `"auto"`

See [langchain-ai/langchain#37178](https://github.com/langchain-ai/langchain/issues/37178).

## License

MIT — see [LICENSE](LICENSE).
