"""
PDF ingestion: parse → chunk by page → embed → store in ChromaDB.
"""

import os
import pdfplumber
import chromadb
from openai import OpenAI

_CHROMA_PATH = "chroma_db"
_COLLECTION = "reports"
_EMBED_MODEL = "text-embedding-3-small"
_MAX_TOKENS_PER_CHUNK = 2000  # approx chars; split long pages in half

_client: OpenAI | None = None
_chroma: chromadb.Collection | None = None


def _get_openai() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def _get_collection() -> chromadb.Collection:
    global _chroma
    if _chroma is None:
        db = chromadb.PersistentClient(path=_CHROMA_PATH)
        _chroma = db.get_or_create_collection(_COLLECTION)
    return _chroma


def _embed(texts: list[str]) -> list[list[float]]:
    resp = _get_openai().embeddings.create(model=_EMBED_MODEL, input=texts)
    return [r.embedding for r in resp.data]


def _doc_name(pdf_path: str) -> str:
    return os.path.splitext(os.path.basename(pdf_path))[0]


def _table_to_text(table: list[list]) -> str:
    """Convert a pdfplumber table (list of rows) to pipe-separated readable text."""
    if not table:
        return ""
    headers = [str(c or "").strip() for c in table[0]]
    rows = []
    for i, row in enumerate(table):
        cells = [str(c or "").strip() for c in row]
        if not any(cells):
            continue
        if i == 0:
            continue  # skip header row; used as column labels below
        if any(headers):
            parts = [f"{h}: {c}" for h, c in zip(headers, cells) if h and c]
        else:
            parts = [c for c in cells if c]
        if parts:
            rows.append(" | ".join(parts))
    return "\n".join(rows)


def is_ingested(pdf_path: str) -> bool:
    """Return True if this PDF has already been indexed."""
    col = _get_collection()
    name = _doc_name(pdf_path)
    results = col.get(where={"doc_name": name}, limit=1)
    return len(results["ids"]) > 0


def ingest(pdf_path: str) -> int:
    """
    Parse PDF, embed each page as a chunk, store in ChromaDB.
    Returns number of chunks added.
    """
    col = _get_collection()
    name = _doc_name(pdf_path)

    chunks = []  # list of (id, text, metadata)

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()

            # Append structured table data (more accurate than plain text for tables)
            tables = page.extract_tables() or []
            table_parts = [_table_to_text(t) for t in tables]
            table_parts = [t for t in table_parts if t]
            if table_parts:
                text = text + "\n\n[TABLE DATA]\n" + "\n\n".join(table_parts)

            if not text:
                continue

            # Split long pages into two halves to stay within token limits
            if len(text) > _MAX_TOKENS_PER_CHUNK:
                mid = len(text) // 2
                blocks = [text[:mid], text[mid:]]
            else:
                blocks = [text]

            for block_idx, block in enumerate(blocks):
                chunk_id = f"{name}_p{page_num}" if len(blocks) == 1 else f"{name}_p{page_num}_b{block_idx}"
                chunks.append((
                    chunk_id,
                    block,
                    {"doc_name": name, "page_num": page_num, "source": os.path.basename(pdf_path)},
                ))

    if not chunks:
        return 0

    # Embed in batches of 100
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        ids = [c[0] for c in batch]
        texts = [c[1] for c in batch]
        metas = [c[2] for c in batch]
        embeddings = _embed(texts)
        col.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metas)

    return len(chunks)


def list_docs() -> list[dict]:
    """Return a list of {doc_name, page_count} for all indexed documents."""
    col = _get_collection()
    all_meta = col.get(include=["metadatas"])["metadatas"]
    counts: dict[str, int] = {}
    for m in all_meta:
        name = m["doc_name"]
        counts[name] = counts.get(name, 0) + 1
    return [{"doc_name": k, "chunks": v} for k, v in sorted(counts.items())]


def delete_doc(doc_name: str) -> None:
    """Remove all chunks for a given document."""
    col = _get_collection()
    col.delete(where={"doc_name": doc_name})
