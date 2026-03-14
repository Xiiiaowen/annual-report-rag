"""
Build a pre-indexed ChromaDB snapshot for the bundled Apple and HSBC reports.

Run this script locally whenever bundled PDFs or chunking logic change:

    OPENAI_API_KEY=sk-... python scripts/build_snapshot.py

The resulting chroma_db/ is committed to git so Streamlit Cloud cold starts
skip the 20-40 second re-embedding step entirely.
"""

import glob
import os
import shutil
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if not os.environ.get("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable is not set.")
    sys.exit(1)

from rag.ingest import ingest, _CHROMA_PATH

BUNDLED_DIR = os.path.join("data", "reports")

# ── Wipe existing DB ──────────────────────────────────────────────────────────
if os.path.exists(_CHROMA_PATH):
    shutil.rmtree(_CHROMA_PATH)
    print(f"Deleted existing {_CHROMA_PATH}/")

# ── Re-ingest all bundled PDFs ────────────────────────────────────────────────
pdfs = sorted(glob.glob(os.path.join(BUNDLED_DIR, "*.pdf")))
if not pdfs:
    print(f"No PDFs found in {BUNDLED_DIR}/")
    sys.exit(1)

for pdf in pdfs:
    name = os.path.basename(pdf)
    print(f"Ingesting {name}…", end=" ", flush=True)
    n = ingest(pdf)
    print(f"{n} chunks")

print(f"\nSnapshot built in {_CHROMA_PATH}/ — commit it to git.")
