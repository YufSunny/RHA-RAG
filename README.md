# Researcher: An AI Assistant for Research

This project demonstrates how to create an AI assistant for reasoning-heavy tasks using agentic RAG over local and fetched resources.

This is the branch based on LangGraph.

## Overview

The Researcher agent follows a structured reasoning pipeline:

```
User Question → Clarify → Retrieve → Grade → Reason → Verify → Answer
```

1. **Clarify** — Translates the natural language question into goal-driven logical statements.
2. **Retrieve** — Uses semantic search over the local knowledge base to fetch relevant documents.
3. **Grade** — Scores retrieved documents for relevance; rewrites the query if documents are irrelevant.
4. **Reason** — Performs structured logical reasoning using retrieved context and citations.
5. **Verify** — Validates the reasoning chain against standard deduction rules.
6. **Answer** — Generates the final response with explicit citations.

## Architecture

- **Document Processing**: Local resources (PDF, images, HTML, markdown, txt, docx) are pre-processed into plain text. PDFs and images are processed via [GLM-OCR](https://www.z.ai/).
- **Embeddings**: Documents are embedded using [Qwen text embedding v4](https://dashscope.aliyun.com/).
- **Vector Store**: In-memory vector store for semantic retrieval.
- **LLM**: [Claude Sonnet 4.6](https://www.anthropic.com/) via kimi API for reasoning and generation.
- **Orchestration**: [LangGraph](https://langchain-ai.github.io/langgraph/) state graph with conditional edges for agentic retrieval and reasoning.

## Project Structure

```
.
├── code/
│   └── main.ipynb              # Main implementation notebook
├── data/
│   ├── local/                  # Local documents (PDF, images, txt, etc.)
│   └── fetched/                # Web-fetched resources
├── reasoner/
│   ├── trial_prompt.md         # Reasoning pipeline prompts (clarify → solve → verify)
│   ├── writer.md               # Research paper writing prompt
│   └── onlineSearch.md         # Online literature search prompt
└── reference/
    └── buildAgenticRAGwithLangGraph.html  # LangGraph tutorial reference
```

## Setup

### Prerequisites

- Python 3.10+
- API keys for:
  - **Z.ai** (GLM-OCR): `ZAI_API_KEY`
  - **DashScope** (Qwen Embeddings): `QWEN_API_KEY`
  - **Kimi / OpenAI-compatible** (Claude LLM): `OPENAI_API_KEY`

### Installation

```bash
pip install -U langgraph langchain langchain-text-splitters langchain-openai beautifulsoup4 requests pymupdf openai zai
```

### Configuration

The notebook will prompt for API keys on first run. Alternatively, set them as environment variables:

```bash
export ZAI_API_KEY="your-zai-key"
export QWEN_API_KEY="your-qwen-key"
export OPENAI_API_KEY="your-openai-key"
```

## Usage

1. Place documents in `data/local/` (supported: `.txt`, `.md`, `.html`, `.pdf`, `.jpg`, `.png`, `.docx`).
2. Open `code/main.ipynb` in Jupyter.
3. Run all cells to build the agentic RAG graph.
4. Invoke the graph with a research question:

```python
for chunk in graph.stream({"messages": [{"role": "user", "content": "Your research question here"}]}):
    for node, update in chunk.items():
        print("Update from node", node)
        update["messages"][-1].pretty_print()
```

## Supported File Types

| Extension | Processing Method |
|-----------|-------------------|
| `.txt`, `.md` | Direct text read |
| `.html`, `.htm` | BeautifulSoup text extraction |
| `.pdf` | GLM-OCR (page-by-page) |
| `.jpg`, `.png` | GLM-OCR |
| `.docx` | PyMuPDF text extraction |

## Dev Notes

- The reasoning layer is implemented as LangGraph nodes (`clarify_question`, `reason`, `verify`) using prompts from `reasoner/trial_prompt.md`.
- Document grading uses structured output (Pydantic) for binary relevance scoring.
- Query rewriting triggers when retrieved documents are judged irrelevant, looping back to retrieval.
