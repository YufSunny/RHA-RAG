# Researcher

**Reasoning-Heavy Agentic RAG Research Assistant** — a LangGraph-based AI system that retrieves documents from a local knowledge base and performs structured logical reasoning to produce verified, cited answers.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API keys
export ZAI_API_KEY="your-zai-key"       # GLM-OCR
export QWEN_API_KEY="your-qwen-key"     # Embeddings
export OPENAI_API_KEY="your-deepseek-key"  # LLM

# 3. Start web server
python server.py
# Open http://localhost:8000

# Or run CLI pipeline
python run.py   # output logged to run.log
```

## Pipeline

```
User Question
    │
    ▼
clarify          ── Translate NL question into goal-driven logical statements
    │
    ▼
generate_query   ── LLM decides: search knowledge base or answer directly
    │
    ├──[no tool call]── END
    │
    ▼
retrieve         ── Semantic search over local vector store (ToolNode)
    │
    ▼
grade            ── Assess document relevance with structured output
    │
    ▼
reason           ── Build logical proof chain (@cite / @common / @MP / @TA)
    │
    ▼
verify           ── Validate each deduction step against inference rules
    │
    ▼
generate_answer  ── Produce final answer with explicit citations
```

## Architecture

| Component | Technology | Provider |
|-----------|-----------|----------|
| Orchestration | [LangGraph](https://langchain-ai.github.io/langgraph/) | — |
| LLM | `deepseek-v4-pro` via `ChatDeepSeekFixed` | [DeepSeek](https://api.deepseek.com) |
| Embeddings | `text-embedding-v4` (batch ≤10) | [Qwen/DashScope](https://dashscope.aliyun.com) |
| OCR | `glm-ocr` via `ZaiClient` (data URI format) | [Z.ai](https://www.z.ai/) |
| Vector Store | `InMemoryVectorStore` | LangChain |
| PDF Render | PyMuPDF `get_pixmap(dpi=200)` | — |
| Web Server | FastAPI + SSE streaming | — |

### Supported File Types

| Extension | Processing |
|-----------|-----------|
| `.txt`, `.md` | Direct read |
| `.html`, `.htm` | BeautifulSoup text extraction |
| `.pdf` | GLM-OCR (rendered as PNG per-page) |
| `.jpg`, `.png` | GLM-OCR |
| `.docx` | PyMuPDF text extraction |

## Project Structure

```
Researcher/
├── server.py                # FastAPI web server
├── run.py                   # CLI pipeline (output to run.log)
├── researcher/              # Core package
│   ├── llm.py               # ChatDeepSeekFixed, model config
│   ├── pipeline.py          # Document loaders, OCR, embeddings
│   └── graph.py             # LangGraph nodes & assembly
├── static/                  # Frontend assets
├── templates/               # HTML templates
├── uploads/                 # User-uploaded documents
├── data/local/              # Pre-loaded documents
├── reasoner/                # SOPs for reasoning procedure
│   ├── trial_prompt.md      # SOP-001: Clarify → Solve → Verify
│   ├── writer.md            # SOP-002: Research paper writing
│   └── onlineSearch.md      # SOP-003: Online literature search
├── requirements.txt
├── .env.example
├── LICENSE
└── README.md
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web UI |
| `GET` | `/api/status` | System status (ready, docs, chunks, errors) |
| `GET` | `/api/files` | List uploaded documents |
| `POST` | `/api/upload` | Upload documents (multipart) |
| `DELETE` | `/api/files/{name}` | Delete a document |
| `POST` | `/api/reindex` | Force re-index all documents |
| `POST` | `/api/chat` | Ask a question (SSE streaming) |

## DeepSeek V4 Patch

`ChatDeepSeekFixed` patches three incompatibilities with DeepSeek V4 thinking mode:

1. **`reasoning_content` preservation** — required across tool-call round-trips; LangChain strips it
2. **List content serialization** — tool/assistant messages with list-type content must be serialized to strings
3. **`tool_choice` demotion** — thinking mode rejects `{"type":"function",...}`; we force `"auto"`

See: [langchain-ai/langchain#37178](https://github.com/langchain-ai/langchain/issues/37178)

## Dev

```bash
# Local tests (no API keys needed)
mamba activate langchain && python test_local.py

# Full pipeline test (needs all API keys)
mamba activate langchain && python run.py
```

## License

MIT — see [LICENSE](LICENSE).
