"""
run.py — RHA-RAG CLI

Usage:
  python run.py "What is a compact set?"
  python run.py                          (prompts interactively)
  echo "Define continuity" | python run.py  (piped)

All output tee'd to run.log. Same pipeline as the web server.
"""

import sys
import io
import builtins
import os
from pathlib import Path
from datetime import datetime

os.chdir(Path(__file__).parent)

# Load .env file if present.
_env = Path(".env")
if _env.exists():
    for _line in _env.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

# ── Logging: tee to run.log ───────────────────────────────────
LOG_PATH = Path("run.log")
LOG_FH = open(LOG_PATH, "w", encoding="utf-8")
_original_print = builtins.print

def print(*args, flush=True, **kwargs):
    _original_print(*args, flush=flush, **kwargs)
    _original_print(*args, file=LOG_FH, flush=flush, **kwargs)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

print("=" * 60)
print(f"RHA-RAG CLI — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# ── Question ──────────────────────────────────────────────────
if len(sys.argv) > 1:
    question = " ".join(sys.argv[1:])
elif sys.stdin.isatty():
    question = input("Question: ").strip()
    if not question:
        question = "What is a topological space and what are its basic properties?"
else:
    question = sys.stdin.read().strip()

print(f"Question: {question}")

# ── Pipeline (shared with server) ─────────────────────────────
from rha_rag.llm import models, create_llms
from rha_rag.pipeline import load_all_documents, create_vectorstore
from rha_rag.graph import build_graph
from langchain_core.messages import HumanMessage

# Check keys
missing = [m.api_key for _, m in models.items() if not os.environ.get(m.api_key)]
if missing:
    print(f"Missing API keys: {missing}")
    print("Set them in environment and retry.")
    sys.exit(1)

# Init LLMs
print("Initializing LLMs...")
response_model, grader_model = create_llms()
print(f"  LLM: {models['llm'].model_name}")

# Load documents (lazy — uses .ocr.md cache, only OCRs new files)
print("Loading documents...")
DATA_DIR = Path("data/local")
UPLOAD_DIR = Path("uploads")
AUTO_SEED_DIR = Path("data/auto-seed")
dirs = [str(UPLOAD_DIR)] if UPLOAD_DIR.exists() and any(UPLOAD_DIR.iterdir()) else []
if DATA_DIR.exists() and any(DATA_DIR.iterdir()):
    dirs.append(str(DATA_DIR))
if AUTO_SEED_DIR.exists() and any(AUTO_SEED_DIR.iterdir()):
    dirs.append(str(AUTO_SEED_DIR))

if not dirs:
    print("No documents found. Place files in data/auto-seed/, data/local/, or uploads/")
    sys.exit(1)

docs = load_all_documents(
    dirs,
    progress_callback=lambda msg, cur, tot: print(f"  [{cur}/{tot}] {msg}"),
)
print(f"  {len(docs)} documents")

# Chunk + embed
print("Indexing...")
_, retriever, chunk_count = create_vectorstore(docs)
print(f"  {chunk_count} chunks")

# Build graph
print("Building graph...")
graph = build_graph(retriever, response_model, grader_model)
print("  compiled")

# ── Run ──────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("Streaming graph...\n")

for chunk in graph.stream(
    {
        "question": question,
        "history": "",
        "fast_mode": True,
        "messages": [HumanMessage(content=question)],
    }
):
    for node_name, update in chunk.items():
        msg = update["messages"][-1]
        content = getattr(msg, "content", str(msg))
        display = content if len(content) <= 2000 else content[:2000] + f"\n... [{len(content)} total chars]"
        print(f"  ── [{node_name}]")
        for line in display.splitlines():
            print(f"  │ {line}")
        print(f"  ──")

print(f"\nDone. Log: {LOG_PATH.resolve()}")
LOG_FH.close()
