"""
server.py — RHA-RAG Web Interface

FastAPI server providing:
  - Document upload & management
  - Streaming chat with the reasoning pipeline
  - Web UI at http://localhost:8000

Usage:
  mamba activate langchain && python server.py
  (or: uvicorn server:app --host 0.0.0.0 --port 8000)
"""

import os
import sys
import io
import json
import shutil
import asyncio
import logging
import traceback
from pathlib import Path
from datetime import datetime

os.chdir(Path(__file__).parent)

# ── Logging: everything to server.log AND console ────────────
LOG_FORMAT = "%(asctime)s  %(levelname)-5s  %(message)s"
LOG_DATE   = "%H:%M:%S"

# File logger — full detail
file_handler = logging.FileHandler("server.log", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE))

# Console logger
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE))

logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, console_handler], force=True)
log = logging.getLogger("rha_rag")

# Redirect stdout/stderr to UTF-8 so Unicode math symbols work
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Hook uvicorn's logger into ours
for name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
    ulog = logging.getLogger(name)
    ulog.handlers = [file_handler, console_handler]
    ulog.setLevel(logging.DEBUG if "access" not in name else logging.INFO)

log.info("── RHA-RAG server starting ──")

# ── App setup ─────────────────────────────────────────────────
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Ensure directories exist
UPLOAD_DIR = Path("uploads")
DATA_DIR = Path("data/local")
UPLOAD_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# Startup — server starts immediately, no blocking init.
# Indexing happens on the first question (or when user clicks Re-index).
# ═══════════════════════════════════════════════════════════════

# Index dirty flag — set to True when files are uploaded/deleted.
# Only re-index when dirty (or no graph exists yet).
_index_dirty: bool = True


def ensure_index():
    """Build the index if needed. Returns True if ready."""
    global _index_dirty

    if _graph is not None and not _index_dirty:
        log.info(f"Index: CACHED — {_doc_count} docs, {_chunk_count} chunks")
        return True

    if _graph is None:
        log.info("Index: BUILDING (first time)")
    else:
        log.info("Index: REBUILDING (new files uploaded)")

    if _ensure_keys():
        return False

    t0 = datetime.now()
    _init_pipeline()
    _index_dirty = False
    elapsed = (datetime.now() - t0).total_seconds()
    if _graph:
        log.info(f"Index: DONE in {elapsed:.1f}s — {_doc_count} docs, {_chunk_count} chunks")
    else:
        log.error(f"Index: FAILED — {_init_errors}")
    return _graph is not None


from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app):
    """Start immediately, no blocking work."""
    log.info("Server ready: http://localhost:8000  (Ctrl+C to stop)")
    yield
    # Clean shutdown
    log.info("Server shutting down...")
    for handler in logging.getLogger().handlers:
        handler.close()


app = FastAPI(title="RHA-RAG", version="0.1.0", lifespan=lifespan)


# ── Request logging middleware ──────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every HTTP request with method, path, status, and duration."""
    start = datetime.now()
    response = await call_next(request)
    duration_ms = (datetime.now() - start).total_seconds() * 1000
    if request.url.path != "/api/chat":  # chat is SSE, logged separately
        log.info(
            f"REQ {request.method:6s} {request.url.path:20s} -> {response.status_code} "
            f"({duration_ms:.0f}ms)"
        )
    return response


# ── Global state (lazy init) ──────────────────────────────────
from rha_rag.llm import models, create_llms
from rha_rag.pipeline import load_all_documents, create_vectorstore
from rha_rag.graph import build_graph

_retriever = None
_graph = None
_response_model = None
_grader_model = None
_doc_count = 0
_chunk_count = 0
_init_errors: list[str] = []


def _ensure_keys():
    """Check that required API keys are set. Returns list of missing keys."""
    missing = []
    for _, m in models.items():
        if not os.environ.get(m.api_key):
            missing.append(m.api_key)
    return missing


def _init_pipeline():
    """Initialize LLMs, load all documents, build vector store and graph."""
    global _retriever, _graph, _response_model, _grader_model, _doc_count, _chunk_count, _init_errors

    _init_errors = []
    t0 = datetime.now()

    # Step 1: Check keys
    missing = _ensure_keys()
    if missing:
        _init_errors = [f"Missing API key(s): {', '.join(missing)}"]
        log.warning(f"Keys: MISSING {missing}")
        return False
    log.info(f"Keys: OK ({len(models)} providers)")

    # Step 2: Init LLMs
    try:
        _response_model, _grader_model = create_llms()
        log.info(f"LLMs: OK ({type(_response_model).__name__})")
    except Exception as e:
        _init_errors.append(f"LLM init: {e}")
        log.error(f"LLMs: FAILED — {e}")
        return False

    # Step 3: Scan directories
    dirs = [str(UPLOAD_DIR)]
    if DATA_DIR.exists() and any(DATA_DIR.iterdir()):
        dirs.append(str(DATA_DIR))
    log.info(f"Scan: {len(dirs)} dir(s) — {dirs}")

    # Step 4: Load documents
    try:
        log.info("Loading documents...")
        docs = load_all_documents(
            dirs,
            progress_callback=lambda msg, cur, tot: log.info(
                f"  [{cur}/{tot}] {msg}"
            ),
        )
    except Exception as e:
        _init_errors.append(f"Document loading: {e}")
        log.error(f"Documents: FAILED — {e}\n{traceback.format_exc()}")
        return False

    _doc_count = len(docs)
    if not docs:
        _init_errors.append("No documents found. Upload some files to get started.")
        log.warning("Documents: NONE found")
        _retriever = None
        _graph = None
        return False
    log.info(f"Documents: {_doc_count} loaded")

    # Step 5: Chunk + embed
    try:
        log.info("Chunking & embedding...")
        vectorstore, _retriever, _chunk_count = create_vectorstore(docs)
        log.info(f"Embedding: {_chunk_count} chunks indexed")
    except Exception as e:
        _init_errors.append(f"Vector store: {e}")
        log.error(f"Embedding: FAILED — {e}\n{traceback.format_exc()}")
        return False

    # Step 6: Build graph
    try:
        _graph = build_graph(_retriever, _response_model, _grader_model)
        log.info(f"Graph: compiled OK")
    except Exception as e:
        _init_errors.append(f"Graph: {e}")
        log.error(f"Graph: FAILED — {e}\n{traceback.format_exc()}")
        return False

    elapsed = (datetime.now() - t0).total_seconds()
    log.info(f"Pipeline: READY in {elapsed:.1f}s — {_doc_count} docs, {_chunk_count} chunks, 7 nodes")
    return True


# ═══════════════════════════════════════════════════════════════
# API endpoints
# ═══════════════════════════════════════════════════════════════

@app.get("/api/status")
async def api_status():
    """Return current system status."""
    missing = _ensure_keys()
    return {
        "ready": _graph is not None,
        "documents": _doc_count,
        "chunks": _chunk_count,
        "missing_keys": missing,
        "errors": _init_errors,
    }


@app.get("/api/files")
async def api_files():
    """List all files in uploads/ and data/local/."""
    files = []
    for d in [UPLOAD_DIR, DATA_DIR]:
        if d.exists():
            for f in d.iterdir():
                if f.is_file():
                    files.append({
                        "name": f.name,
                        "size": f.stat().st_size,
                        "dir": str(d),
                        "modified": datetime.fromtimestamp(
                            f.stat().st_mtime
                        ).isoformat(),
                    })
    return sorted(files, key=lambda x: x["name"])


@app.post("/api/upload")
async def api_upload(files: list[UploadFile] = File(...)):
    """Upload one or more documents. Index is rebuilt on next question."""
    saved = []
    for f in files:
        path = UPLOAD_DIR / f.filename
        content = await f.read()
        path.write_bytes(content)
        size_kb = len(content) / 1024
        log.info(f"Upload: {f.filename} ({size_kb:.1f} KB)")
        saved.append(f.filename)

    # Invalidate cached index — will rebuild on next /api/chat
    global _index_dirty
    _index_dirty = True
    log.info(f"Upload: {len(saved)} file(s) saved, index invalidated")

    return {
        "saved": saved,
        "documents": _doc_count,
        "chunks": _chunk_count,
    }


@app.delete("/api/files/{name}")
async def api_delete(name: str):
    """Delete a file from uploads/. Index invalidated, rebuilds on next question."""
    path = UPLOAD_DIR / name
    if path.exists():
        path.unlink()
        global _index_dirty
        _index_dirty = True
        log.info(f"Delete: {name} removed, index invalidated")
        return {"deleted": name}
    log.warning(f"Delete: {name} — not found")
    return JSONResponse({"error": "not found"}, status_code=404)


@app.post("/api/reindex")
async def api_reindex():
    """Force re-index all documents immediately."""
    global _index_dirty
    _index_dirty = True  # Force rebuild
    import asyncio
    success = await asyncio.get_event_loop().run_in_executor(None, ensure_index)
    return {
        "success": success,
        "documents": _doc_count,
        "chunks": _chunk_count,
        "errors": _init_errors,
    }


@app.post("/api/chat")
async def api_chat(request: Request):
    """Stream the graph execution as SSE events.

    On first call, builds the document index (lazy init).
    Each event is JSON with: {node, content, done}
    """
    t0 = datetime.now()
    body = await request.json()
    question = body.get("question", "").strip()
    log.info(f"Chat: question=\"{question[:80]}{'...' if len(question) > 80 else ''}\"")
    if not question:
        return JSONResponse({"error": "question is required"}, status_code=400)

    # Lazy-init on first question (or when files changed)
    try:
        ready = await asyncio.get_event_loop().run_in_executor(None, ensure_index)
    except Exception as e:
        log.error(f"Chat: indexing failed — {e}")
        return JSONResponse({
            "error": f"Indexing failed: {e}",
            "details": _init_errors,
        }, status_code=503)

    if not ready:
        log.warning("Chat: not ready — no documents indexed")
        return JSONResponse({
            "error": "Pipeline not ready. Upload documents first.",
            "details": _init_errors,
        }, status_code=503)

    async def generate():
        loop = asyncio.get_event_loop()

        # Send indexing status
        yield f"data: {json.dumps({'node': 'status', 'content': f'{_doc_count} documents, {_chunk_count} chunks', 'done': False})}\n\n"

        def run_graph():
            results = []
            for chunk in _graph.stream(
                {"messages": [{"role": "user", "content": question}]}
            ):
                for node_name, update in chunk.items():
                    msg = update["messages"][-1]
                    content = getattr(msg, "content", str(msg))
                    log.info(f"  Node [{node_name}]: {len(content)} chars")
                    results.append((node_name, content))
            return results

        try:
            results = await loop.run_in_executor(None, run_graph)
            for node_name, content in results:
                event = json.dumps({
                    "node": node_name,
                    "content": content,
                    "done": False,
                }, ensure_ascii=False)
                yield f"data: {event}\n\n"

            elapsed = (datetime.now() - t0).total_seconds()
            nodes_seen = [n for n, _ in results]
            log.info(f"Chat: DONE in {elapsed:.1f}s — nodes: {nodes_seen}")
            yield f"data: {json.dumps({'node': 'done', 'content': '', 'done': True})}\n\n"

        except Exception as e:
            log.error(f"Chat: graph error — {e}\n{traceback.format_exc()}")
            error_event = json.dumps({
                "node": "error",
                "content": str(e),
                "done": True,
            })
            yield f"data: {error_event}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ═══════════════════════════════════════════════════════════════
# Frontend
# ═══════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def index():
    return Path("templates/index.html").read_text(encoding="utf-8")


# HTML now served from templates/index.html


if __name__ == "__main__":
    import signal as _signal

    def _force_quit(sig, frame):
        log.info("Shutdown signal received.")
        for h in logging.getLogger().handlers:
            h.close()
        os._exit(0)

    _signal.signal(_signal.SIGINT, _force_quit)
    _signal.signal(_signal.SIGTERM, _force_quit)

    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, log_config=None)
    except KeyboardInterrupt:
        _force_quit(None, None)
