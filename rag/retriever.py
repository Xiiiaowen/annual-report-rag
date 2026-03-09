"""
Query ChromaDB for the top-k most relevant chunks.
"""

import os
from openai import OpenAI
import chromadb

from .ingest import _get_collection, _EMBED_MODEL

_client: OpenAI | None = None


def _get_openai() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def retrieve(query: str, k: int = 5) -> list[dict]:
    """
    Embed the query and return the top-k most similar chunks.
    Each result: {text, doc_name, page_num, source, score}
    """
    resp = _get_openai().embeddings.create(model=_EMBED_MODEL, input=[query])
    query_embedding = resp.data[0].embedding

    col = _get_collection()
    results = col.query(
        query_embeddings=[query_embedding],
        n_results=min(k, col.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for text, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": text,
            "doc_name": meta["doc_name"],
            "page_num": meta["page_num"],
            "source": meta["source"],
            "score": round(1 - distance, 3),  # cosine similarity
        })

    return chunks
