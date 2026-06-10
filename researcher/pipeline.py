"""Document loading, OCR, embeddings, and vector store."""

import os
import base64
import bs4
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from researcher.llm import models


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


def ocr_image(file_path: str) -> str:
    """OCR a JPG/PNG via data URI. Returns markdown text."""
    data = Path(file_path).read_bytes()
    ext = Path(file_path).suffix.lower()
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png"}[ext.lstrip(".")]
    b64 = base64.b64encode(data).decode("ascii")
    resp = _ocr_client().layout_parsing.create(
        model="glm-ocr",
        file=f"data:image/{mime};base64,{b64}",
    )
    return resp.md_results


def ocr_pdf(
    path: str,
    max_pages: Optional[int] = None,
    dpi: int = 200,
    progress_callback=None,
) -> str:
    """Render each PDF page as PNG, OCR individually. Returns combined markdown.

    Args:
        path: Path to PDF file.
        max_pages: Max pages to process (None = all).
        dpi: Render resolution.
        progress_callback: Called with (current, total) for progress reporting.
    """
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
    return "\n\n".join(results)


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


def create_vectorstore(documents: list[Document]):
    """Create an in-memory vector store from documents.
    Returns (vectorstore, retriever, chunk_count).
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.vectorstores import InMemoryVectorStore

    if not documents:
        return None, None, 0

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512, chunk_overlap=128
    )
    splits = splitter.split_documents(documents)

    vectorstore = InMemoryVectorStore.from_documents(splits, embedding=CustomEmbed())
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return vectorstore, retriever, len(splits)
