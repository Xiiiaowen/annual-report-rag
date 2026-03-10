"""
Build a prompt from retrieved chunks and call GPT-4o-mini for an answer.
"""

import os
from openai import OpenAI

_client: OpenAI | None = None
_MODEL = "gpt-4o-mini"


def _get_openai() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def answer(question: str, chunks: list[dict], history: list[dict] | None = None) -> dict:
    """
    Generate an answer grounded in the retrieved chunks.
    history: list of {"role": "user"|"assistant", "content": str} from previous turns.
    Returns {answer: str, sources: list[dict]}
    """
    if not chunks:
        return {
            "answer": "No relevant content found in the indexed documents.",
            "sources": [],
        }

    # Build context block
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"[{i}] Source: {chunk['doc_name']}, page {chunk['page_num']}\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    system_prompt = (
        "You are a financial analyst assistant. Answer the user's question using ONLY "
        "the provided document excerpts. If the answer is not contained in the excerpts, "
        "say so clearly — do not guess or use outside knowledge. "
        "After your answer, list the sources you used in the format: "
        "[doc_name, p.PAGE_NUM]"
    )

    user_prompt = f"Question: {question}\n\nDocument excerpts:\n\n{context}"

    # Build messages: system + last 3 exchanges of history + current question+context
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        for msg in history[-6:]:  # last 3 exchanges (6 messages)
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_prompt})

    response = _get_openai().chat.completions.create(
        model=_MODEL,
        messages=messages,
        temperature=0,
    )

    answer_text = response.choices[0].message.content.strip()

    # Deduplicate sources (same doc+page may appear multiple times from split chunks)
    seen = set()
    sources = []
    for chunk in chunks:
        key = (chunk["doc_name"], chunk["page_num"])
        if key not in seen:
            seen.add(key)
            sources.append({
                "doc_name": chunk["doc_name"],
                "page_num": chunk["page_num"],
                "excerpt": chunk["text"][:300] + ("..." if len(chunk["text"]) > 300 else ""),
            })

    return {"answer": answer_text, "sources": sources}
