# CLAUDE.md — annual-report-rag

## What This Project Is

A RAG (Retrieval-Augmented Generation) app that lets users ask natural-language questions
about company annual reports and get answers with page-level citations.

The project demonstrates a different AI pattern from mdm-agent: instead of tool-use/agentic
loops, this uses embeddings + vector search to ground LLM answers in real documents.

## User-Facing Behaviour

1. App opens with 2 pre-loaded annual reports already indexed (Apple 2025, HSBC 2025)
2. User can upload additional PDF reports via the sidebar
3. User types a question in the main chat area
4. App retrieves the most relevant passages from all indexed documents
5. LLM answers the question using those passages, and shows which doc + page each fact came from

Example questions the demo should handle well:
- "What was Apple's total revenue last fiscal year?"
- "What risks did HSBC highlight in their latest annual report?"
- "How does HSBC describe its climate strategy?"

## Project Structure

```
annual-report-rag/
├── app.py                  # Streamlit UI
├── rag/
│   ├── __init__.py
│   ├── ingest.py           # PDF parsing, chunking, embedding, ChromaDB storage
│   ├── retriever.py        # Query ChromaDB, return top-k chunks with metadata
│   └── answerer.py         # Build prompt, call GPT-4o-mini, return answer + citations
├── data/
│   └── reports/            # Pre-loaded PDFs (committed to repo)
│       ├── apple-2025.pdf
│       └── hsbc-2025.pdf
├── chroma_db/              # Vector store — gitignored, rebuilt on cold start
├── .env
├── .env.example
├── requirements.txt
├── README.md
└── CLAUDE.md
```

## Tech Stack

| Layer | Library | Notes |
|---|---|---|
| PDF parsing | pdfplumber | Preserves layout; extract text page by page |
| Embeddings | openai text-embedding-3-small | Cheap (~$0.02/1M tokens), good quality |
| Vector store | chromadb | Local, no account needed |
| LLM | gpt-4o-mini | Same as other projects in portfolio |
| UI | streamlit | Same as other projects in portfolio |

## Chunking Strategy (important — not naive fixed-size)

Each PDF is split **page by page**. Each page = one chunk.
Each chunk stored with metadata:
- `doc_name`: filename without extension (e.g. "apple-2023")
- `page_num`: 1-indexed page number
- `text`: the full page text

Rationale: annual reports have natural page boundaries; page-level chunks give precise citations
and avoid splitting mid-sentence across arbitrary character counts.

If a page is very long (>2000 tokens), split it into two half-page sub-chunks, both tagged
with the same page_num.

## ChromaDB Collection

Single collection named `"reports"`.
Each chunk stored as a ChromaDB document with:
- `id`: `"{doc_name}_p{page_num}"` (or `"{doc_name}_p{page_num}_b{block}"` for split pages)
- `document`: the chunk text
- `metadata`: `{"doc_name": ..., "page_num": ..., "source": ...}`

On startup:
1. Check if collection already contains documents for each bundled report (by doc_name)
2. If missing, ingest that report automatically
3. User uploads always trigger ingestion immediately after upload

## Retrieval

Query: top-5 chunks by cosine similarity (ChromaDB default).
No re-ranking for now — keep it simple.

Return each chunk with its metadata so the answer layer can cite sources.

## Answer Generation

System prompt instructs the LLM to:
- Answer using ONLY the provided context passages
- If the answer is not in the context, say so explicitly (no hallucination)
- After the answer, list the sources used as: `[doc_name, p.{page_num}]`

Response displayed in two parts:
1. The answer text
2. An expandable "Sources" section listing each cited chunk with a short excerpt

## UI Layout

Sidebar:
- List of currently indexed documents (name + page count)
- PDF uploader (accepts multiple files)
- "Clear uploaded docs" button (removes user uploads, keeps bundled ones)

Main area:
- Question input box at the top
- Answer + sources displayed below
- No chat history needed — single Q&A per query is fine

## API Keys Required

- `OPENAI_API_KEY` — embeddings + LLM

No other keys needed. ChromaDB is local. PDFs are bundled.

## What to Gitignore

- `.env`
- `chroma_db/` is intentionally committed — it contains the pre-built snapshot for bundled reports
- To rebuild the snapshot: `OPENAI_API_KEY=sk-... python scripts/build_snapshot.py` then commit chroma_db/

## Code Style

- Follow the same patterns as mdm-agent: simple functions, no unnecessary classes
- Each file in `rag/` has one clear responsibility
- No LangChain or LlamaIndex — implement RAG directly so the logic is visible and learnable
- Keep functions short and readable

## What This Demonstrates (for portfolio/interviews)

- RAG pipeline built from scratch (not a framework wrapper)
- Embedding and vector search concepts understood, not just used
- Section-aware chunking with metadata (not naive fixed-size)
- Citation/grounding — answers traceable to specific pages
- Multi-document retrieval across multiple uploaded reports
- Cost-aware: text-embedding-3-small is cheap; bundled docs only embedded once
