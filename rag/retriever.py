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


def _embed_query(query: str) -> list[float]:
    resp = _get_openai().embeddings.create(model=_EMBED_MODEL, input=[query])
    return resp.data[0].embedding


MIN_SCORE = 0.30  # chunks below this cosine similarity are considered off-topic

_REWRITE_SYSTEM = (
    "You are a search query optimizer for financial annual reports. "
    "Rewrite the user's question into the best possible search query for retrieving "
    "relevant passages from an annual report vector store.\n\n"
    "Rules:\n"
    "- Expand financial terminology to include synonyms used in annual reports "
    "(e.g. 'net income' → 'net income profit for the year net profit earnings attributable').\n"
    "- If the question is a follow-up that uses pronouns or refers to previous context "
    "(e.g. 'What about the year before?', 'How did it change?'), resolve it using the "
    "conversation history to produce a fully self-contained query.\n"
    "- Return ONLY the rewritten query — no explanation, no punctuation at the end."
)


def rewrite_query(question: str, history: list[dict] | None = None) -> str:
    """
    Use GPT-4o-mini to rewrite the question into an optimised retrieval query.
    Expands financial synonyms and resolves follow-up references from history.
    Returns the rewritten query string (falls back to original on any error).
    """
    messages = [{"role": "system", "content": _REWRITE_SYSTEM}]
    if history:
        for msg in history[-4:]:  # last 2 exchanges for context
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})

    try:
        resp = _get_openai().chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
            max_tokens=100,
        )
        rewritten = resp.choices[0].message.content.strip()
        return rewritten if rewritten else question
    except Exception:
        return question


def retrieve(query: str, k: int = 5) -> list[dict]:
    """
    Embed the query and return the top-k most similar chunks across all docs.
    Chunks with score < MIN_SCORE are dropped; returns empty list if none pass.
    Each result: {text, doc_name, page_num, source, score}
    """
    query_embedding = _embed_query(query)
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
            "score": round(1 - distance, 3),
        })

    return [c for c in chunks if c["score"] >= MIN_SCORE]


def retrieve_per_doc(query: str, doc_names: list[str], k_per_doc: int = 3) -> list[dict]:
    """
    For comparison queries: retrieve top-k chunks from each document independently,
    guaranteeing representation from every document.
    Chunks with score < MIN_SCORE are dropped.
    """
    query_embedding = _embed_query(query)
    col = _get_collection()
    chunks = []
    for doc_name in doc_names:
        results = col.query(
            query_embeddings=[query_embedding],
            n_results=k_per_doc,
            where={"doc_name": doc_name},
            include=["documents", "metadatas", "distances"],
        )
        doc_chunks = []
        for text, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            doc_chunks.append({
                "text": text,
                "doc_name": meta["doc_name"],
                "page_num": meta["page_num"],
                "source": meta["source"],
                "score": round(1 - distance, 3),
            })
        chunks.extend(c for c in doc_chunks if c["score"] >= MIN_SCORE)
    return chunks


def is_comparison_query(question: str, doc_names: list[str]) -> bool:
    """
    Return True if the question appears to compare multiple indexed documents.
    Triggers when 2+ doc base-names are mentioned, or when comparison keywords
    appear alongside at least one doc mention.
    """
    q = question.lower()
    # Extract company-level base names: "apple-2025" → "apple"
    bases = [name.split("-")[0].lower() for name in doc_names]
    matched = [b for b in bases if b in q]
    if len(matched) >= 2:
        return True
    comparison_words = {"compare", "differ", "difference", "versus", " vs ", "both", "between", "contrast"}
    if any(w in q for w in comparison_words) and len(matched) >= 1:
        return True
    return False
