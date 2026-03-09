"""
Annual Report Q&A — Streamlit UI
"""

import os
import glob
import tempfile

import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

# Streamlit Cloud: inject secrets into os.environ so rag/ modules can read them
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

from rag.ingest import ingest, is_ingested, list_docs, delete_doc
from rag.retriever import retrieve
from rag.answerer import answer

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Annual Report Q&A", page_icon="📄", layout="wide")

# ── Auto-ingest bundled reports on startup ───────────────────────────────────
BUNDLED_DIR = os.path.join(os.path.dirname(__file__), "data", "reports")

def _ingest_bundled():
    pdfs = glob.glob(os.path.join(BUNDLED_DIR, "*.pdf"))
    pending = [p for p in pdfs if not is_ingested(p)]
    if not pending:
        return

    st.info(
        "**First visitor since the app last slept — indexing reports now.**\n\n"
        "Streamlit Cloud puts apps to sleep when idle. On wake-up, the vector store is "
        "rebuilt by splitting each PDF into pages, embedding them with OpenAI, and storing "
        "them in ChromaDB. This takes ~20–40 seconds and happens once per session."
    )

    progress = st.progress(0, text="Starting…")
    total = len(pending)

    for i, pdf in enumerate(pending):
        name = os.path.basename(pdf)
        progress.progress((i) / total, text=f"Step {i+1}/{total} — Parsing and embedding **{name}**…")
        n = ingest(pdf)
        progress.progress((i + 1) / total, text=f"✅ {name} indexed ({n} pages)")

    progress.empty()
    st.rerun()

_ingest_bundled()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 Annual Report Q&A")
    st.caption("Ask questions across indexed annual reports. Answers include page citations.")

    st.divider()

    # Indexed documents
    st.subheader("Indexed documents")
    docs = list_docs()
    if docs:
        for doc in docs:
            st.markdown(f"- **{doc['doc_name']}** ({doc['chunks']} chunks)")
    else:
        st.caption("No documents indexed yet.")

    st.divider()

    # Upload new PDFs
    st.subheader("Upload a report")
    uploaded = st.file_uploader("Add a PDF annual report", type="pdf", accept_multiple_files=True)
    if uploaded:
        for f in uploaded:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name
            # Use original filename as doc name
            dest = os.path.join(BUNDLED_DIR, f.name)
            os.replace(tmp_path, dest)
            if not is_ingested(dest):
                with st.spinner(f"Indexing {f.name}…"):
                    n = ingest(dest)
                st.toast(f"Indexed {f.name} ({n} chunks)", icon="✅")
                st.rerun()

    st.divider()

    # Delete uploaded (non-bundled) docs
    bundled_names = {
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(BUNDLED_DIR, "*.pdf"))
        if p in [os.path.join(BUNDLED_DIR, pdf) for pdf in os.listdir(BUNDLED_DIR)]
    }
    # Simpler: list docs that aren't in the bundled set
    bundled_basenames = {
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(BUNDLED_DIR, "apple-*.pdf"))
        + glob.glob(os.path.join(BUNDLED_DIR, "hsbc-*.pdf"))
    }
    user_docs = [d for d in docs if d["doc_name"] not in bundled_basenames]
    if user_docs:
        st.subheader("Remove uploaded reports")
        for doc in user_docs:
            if st.button(f"Remove {doc['doc_name']}", key=f"del_{doc['doc_name']}"):
                delete_doc(doc["doc_name"])
                # Also delete the file
                pdf_path = os.path.join(BUNDLED_DIR, doc["doc_name"] + ".pdf")
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)
                st.rerun()

    st.divider()
    st.caption("Powered by GPT-4o-mini + ChromaDB + pdfplumber")

# ── Main area ─────────────────────────────────────────────────────────────────
st.header("Ask a question")

if not docs:
    st.info("No documents indexed yet. Upload a PDF in the sidebar to get started.")
    st.stop()

question = st.text_input(
    "Question",
    placeholder="e.g. What was Apple's total revenue last fiscal year?",
    label_visibility="collapsed",
)

if question:
    with st.spinner("Searching and generating answer…"):
        chunks = retrieve(question, k=5)
        result = answer(question, chunks)

    st.markdown("### Answer")
    st.write(result["answer"])

    if result["sources"]:
        st.markdown("### Sources")
        for src in result["sources"]:
            with st.expander(f"📄 {src['doc_name']}  —  page {src['page_num']}"):
                st.caption(src["excerpt"])
