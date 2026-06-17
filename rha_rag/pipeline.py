"""Document loading, OCR, embeddings, and vector store."""

import os
import base64
import bs4
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from rha_rag.llm import models


# ═══════════════════════════════════════════════════════════════
# GLM-OCR
# ═══════════════════════════════════════════════════════════════

_ocr_client_cache = None


def _ocr_client():
    global _ocr_client_cache
    if _ocr_client_cache is not None:
        return _ocr_client_cache
    from zai import ZaiClient
    _ocr_client_cache = ZaiClient(api_key=os.environ["ZAI_API_KEY"])
    return _ocr_client_cache


def _ocr_cache_path(file_path: str) -> Path:
    """Return the .ocr.md cache path for a given file."""
    p = Path(file_path)
    return p.with_name(p.name + ".ocr.md")


def _read_cache(file_path: str) -> str | None:
    """Return cached OCR text if newer than the source file. None otherwise."""
    cache = _ocr_cache_path(file_path)
    if cache.exists() and cache.stat().st_mtime >= Path(file_path).stat().st_mtime:
        return cache.read_text(encoding="utf-8")
    return None


def _write_cache(file_path: str, text: str):
    """Save OCR result to cache file."""
    _ocr_cache_path(file_path).write_text(text, encoding="utf-8")


def ocr_image(file_path: str) -> str:
    """OCR a JPG/PNG via data URI. Returns markdown text. Cached to .ocr.md."""
    cached = _read_cache(file_path)
    if cached is not None:
        return cached
    data = Path(file_path).read_bytes()
    ext = Path(file_path).suffix.lower()
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png"}[ext.lstrip(".")]
    b64 = base64.b64encode(data).decode("ascii")
    resp = _ocr_client().layout_parsing.create(
        model="glm-ocr",
        file=f"data:image/{mime};base64,{b64}",
    )
    _write_cache(file_path, resp.md_results)
    return resp.md_results


def ocr_pdf(
    path: str,
    max_pages: Optional[int] = None,
    dpi: int = 200,
    progress_callback=None,
) -> str:
    """Render each PDF page as PNG, OCR individually. Cached to .ocr.md.

    Args:
        path: Path to PDF file.
        max_pages: Max pages to process (None = all).
        dpi: Render resolution.
        progress_callback: Called with (current, total) for progress reporting.
    """
    cached = _read_cache(path)
    if cached is not None:
        return cached

    import fitz
    doc = fitz.open(path)
    total = min(len(doc), max_pages) if max_pages else len(doc)
    results = []
    for i in range(total):
        pix = doc[i].get_pixmap(dpi=dpi)
        img_bytes = pix.tobytes("png")
        b64 = base64.b64encode(img_bytes).decode("ascii")
        resp = _ocr_client().layout_parsing.create(
            model="glm-ocr",
            file=f"data:image/png;base64,{b64}",
        )
        results.append(resp.md_results)
        if progress_callback:
            progress_callback(i + 1, total)
    doc.close()
    text = "\n\n".join(results)
    _write_cache(path, text)
    return text


# ═══════════════════════════════════════════════════════════════
# File loaders
# ═══════════════════════════════════════════════════════════════

SUPPORTED = (".txt", ".md", ".html", ".htm", ".pdf", ".jpg", ".jpeg", ".png", ".docx")


def _safe_path(path: str) -> str:
    """Convert a filesystem path to forward-slash form for safe display.

    On Windows, backslashes in paths (e.g. ``data\\local\\rudin.pdf``)
    contain escape sequences like ``\\r`` (carriage return) and ``\\t``
    (tab) that corrupt output when printed.  Always store/display paths
    with forward slashes.
    """
    return Path(path).as_posix()


def load_text(path: str) -> list[Document]:
    text = Path(path).read_text(encoding="utf-8")
    return [Document(page_content=text, metadata={"source": _safe_path(path), "name": Path(path).name})]


def load_html(path: str) -> list[Document]:
    text = Path(path).read_text(encoding="utf-8")
    soup = bs4.BeautifulSoup(text, "html.parser")
    for s in soup(["script", "style"]):
        s.decompose()
    text = soup.get_text(separator="\n")
    lines = (l.strip() for l in text.splitlines())
    text = "\n".join(l for l in lines if l)
    return [Document(page_content=text, metadata={"source": _safe_path(path), "name": Path(path).name})]


def load_docx(path: str) -> list[Document]:
    import fitz
    doc = fitz.open(path)
    text = "\n\n".join(pg.get_text() for pg in doc)
    doc.close()
    return [Document(page_content=text, metadata={"source": _safe_path(path), "name": Path(path).name})]


def load_pdf(path: str, progress_callback=None) -> list[Document]:
    import fitz
    p = Path(path)
    src = fitz.open(path)
    n_pages = len(src)
    src.close()
    text = ocr_pdf(path, progress_callback=progress_callback)
    return [Document(page_content=text, metadata={"source": _safe_path(path), "name": p.name, "pages": n_pages})]


def load_image(path: str) -> list[Document]:
    p = Path(path)
    text = ocr_image(path)
    return [Document(page_content=text, metadata={"source": _safe_path(path), "name": p.name})]


def discover_files(data_dir: str) -> list[str]:
    """Auto-discover supported files in a directory. Returns forward-slash paths."""
    p = Path(data_dir)
    if not p.exists():
        return []
    return sorted(
        f.as_posix() for f in p.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED
    )


def load_all_documents(
    data_dirs: list[str],
    progress_callback=None,
) -> list[Document]:
    """Load all documents from multiple directories.

    Args:
        data_dirs: List of directory paths to scan.
        progress_callback: Called with (status_msg, current, total).
                           status_msg is e.g. "Loading calculus_notes.docx" or
                           "OCR page 5/15: rudin.pdf".

    Returns:
        List of Document objects.
    """
    all_files = []
    for d in data_dirs:
        all_files.extend(discover_files(d))

    docs = []
    for idx, fp in enumerate(all_files):
        ext = Path(fp).suffix.lower()
        name = Path(fp).name
        try:
            if progress_callback:
                progress_callback(f"Loading {name}", idx + 1, len(all_files))

            if ext in (".txt", ".md"):
                docs.extend(load_text(fp))
            elif ext in (".html", ".htm"):
                docs.extend(load_html(fp))
            elif ext == ".docx":
                docs.extend(load_docx(fp))
            elif ext == ".pdf":

                def pdf_progress(current, total):
                    if progress_callback:
                        progress_callback(
                            f"OCR page {current}/{total}: {name}",
                            idx + 1, len(all_files),
                        )

                docs.extend(load_pdf(fp, progress_callback=pdf_progress))
                if progress_callback:
                    progress_callback(
                        f"Loaded {name}", idx + 1, len(all_files),
                    )
            elif ext in (".jpg", ".jpeg", ".png"):
                docs.extend(load_image(fp))
        except Exception as e:
            if progress_callback:
                progress_callback(f"Error: {name} - {e}", idx + 1, len(all_files))

    return docs


# ═══════════════════════════════════════════════════════════════
# Embeddings
# ═══════════════════════════════════════════════════════════════

from langchain_core.embeddings import Embeddings
from openai import OpenAI


class CustomEmbed(Embeddings):
    """Qwen text-embedding-v4 wrapper. Batch size limited to 10."""

    def __init__(self, model: str | None = None):
        self.model = model or models["embed"].model_name
        self.client = OpenAI(
            api_key=os.environ.get(models["embed"].api_key),
            base_url=models["embed"].base_url,
            max_retries=5,
            timeout=60,
        ).embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        all_embeddings = []
        for i in range(0, len(texts), 10):
            batch = texts[i : i + 10]
            res = self.client.create(model=self.model, input=batch)
            all_embeddings.extend(d.embedding for d in res.data)
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        res = self.client.create(model=self.model, input=[text])
        return res.data[0].embedding


MILVUS_URI = "milvus.db"  # local file store via milvus-lite (no server needed)


# ═══════════════════════════════════════════════════════════════
# Vector store — thin MilvusClient wrapper (no ORM path)
#
# langchain-milvus 0.3.3 reaches the legacy pymilvus ORM ``Collection`` API
# internally, which is incompatible with pymilvus 2.6.x + milvus-lite 3.0
# (MilvusClient registers under a generated alias absent from the ORM
# ``connections`` registry). This wrapper talks to ``MilvusClient`` directly,
# sidestepping the ORM entirely.
# ═══════════════════════════════════════════════════════════════

from typing import Any
from langchain_core.retrievers import BaseRetriever


class _MilvusRetriever(BaseRetriever):
    """LangChain retriever backed by a ``MilvusLiteStore``."""

    store: Any
    k: int = 5

    def _get_relevant_documents(self, query, *, run_manager=None):  # type: ignore[override]
        return self.store.similarity_search(query, k=self.k)


def _try_wipe(uri: str) -> bool:
    """Remove the milvus-lite db at ``uri`` if possible.

    Returns True if the db is gone (or never existed). Returns False if it is
    locked by another client in this process — milvus-lite holds a file lock
    for the life of the process that ``close()`` does not release, so a wipe
    only succeeds when no client is open yet (fresh process / first build).
    """
    import shutil

    p = Path(uri)
    if not p.exists():
        return True
    try:
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
        return True
    except OSError:
        return False


class MilvusLiteStore:
    """Minimal Milvus vector store over ``pymilvus.MilvusClient`` (milvus-lite).

    Collection schema: pk (INT64, auto_id), text (VARCHAR), vector
    (FLOAT_VECTOR), metadata (JSON — preserves source/name/pages). Metadata is
    stored as a single JSON field so arbitrary per-chunk metadata round-trips
    without dynamic-field output quirks.

    Re-indexing strategy: with ``drop_old=True`` (the default, matching the
    full-reindex behaviour) the whole ``milvus.db`` is wiped when no client
    holds it; otherwise a uniquely-named collection is used. This avoids both
    milvus-lite's process-lifetime file lock and its buggy ``drop_collection``
    on Windows (``WinError 183`` renaming ``manifest.json``).
    """

    def __init__(
        self,
        embedding: "CustomEmbed",
        uri: str = MILVUS_URI,
        collection_name: str = "rha_rag",
        drop_old: bool = True,
    ):
        from pymilvus import MilvusClient

        self._embedding = embedding
        self._uri = uri

        if drop_old:
            if _try_wipe(uri):
                self._collection = collection_name
            else:
                import uuid

                self._collection = f"{collection_name}_{uuid.uuid4().hex[:8]}"
        else:
            self._collection = collection_name

        self._client = MilvusClient(uri=uri)

        # Determine vector dimension from a probe embedding.
        self._dim = len(embedding.embed_query("dimension probe"))

        if not self._client.has_collection(self._collection):
            self._create_collection()

    def _create_collection(self):
        from pymilvus import DataType

        schema = self._client.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field("pk", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self._dim)
        schema.add_field("metadata", DataType.JSON)

        self._client.create_collection(self._collection, schema=schema)
        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="vector", index_type="AUTOINDEX", metric_type="COSINE"
        )
        self._client.create_index(self._collection, index_params=index_params)
        self._client.load_collection(self._collection)

    def add_documents(self, documents: list[Document]) -> None:
        """Embed (batched) and insert documents into the collection."""
        if not documents:
            return
        texts = [d.page_content for d in documents]
        vectors = self._embedding.embed_documents(texts)
        rows = [
            {"text": d.page_content, "vector": vec, "metadata": dict(d.metadata)}
            for d, vec in zip(documents, vectors)
        ]
        self._client.insert(self._collection, rows)

    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        """Return the k most similar documents to ``query``."""
        vec = self._embedding.embed_query(query)
        res = self._client.search(
            self._collection,
            data=[vec],
            limit=k,
            output_fields=["text", "metadata"],
        )
        docs = []
        for hit in res[0]:
            entity = hit.get("entity", {})
            text = entity.get("text", "")
            meta = entity.get("metadata") or {}
            docs.append(Document(page_content=text, metadata=meta))
        return docs

    def as_retriever(self, search_kwargs: dict | None = None):
        k = (search_kwargs or {}).get("k", 5)
        return _MilvusRetriever(store=self, k=k)


def create_vectorstore(documents: list[Document]):
    """Create a Milvus vector store from documents.
    Returns (vectorstore, retriever, chunk_count).

    Uses milvus-lite with a local file (``milvus.db``), so no Milvus server is
    required. ``drop_old=True`` rebuilds the collection from scratch on every
    call, matching the existing full-reindex behaviour.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    if not documents:
        return None, None, 0

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512, chunk_overlap=128
    )
    splits = splitter.split_documents(documents)

    vectorstore = MilvusLiteStore(CustomEmbed(), drop_old=True)
    vectorstore.add_documents(splits)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return vectorstore, retriever, len(splits)
