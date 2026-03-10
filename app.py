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
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass  # No secrets.toml locally — that's fine, .env is used instead

from rag.ingest import ingest, is_ingested, list_docs, delete_doc
from rag.retriever import retrieve
from rag.answerer import answer

# ── Constants ─────────────────────────────────────────────────────────────────
BUNDLED_DIR = os.path.join(os.path.dirname(__file__), "data", "reports")
MAX_UPLOAD_MB = 10
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Annual Report Q&A", page_icon="📄", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 Annual Report Q&A")

    # ── Indexed documents ─────────────────────────────────────────────────────
    st.subheader("Indexed documents")
    docs = list_docs()
    if docs:
        for doc in docs:
            st.markdown(f"- **{doc['doc_name']}** ({doc['chunks']} chunks)")
    else:
        st.caption("No documents indexed yet.")

    st.divider()

    # ── Upload new PDFs ───────────────────────────────────────────────────────
    with st.expander("🔑 OpenAI API key (optional)"):
        user_key = st.text_input(
            "OpenAI API key",
            type="password",
            placeholder="sk-… (uses shared key if blank)",
            label_visibility="collapsed",
        )
        if user_key.strip():
            os.environ["OPENAI_API_KEY"] = user_key.strip()
            st.success("Using your API key.", icon="✅")

    with st.expander("📂 Upload a report"):
        st.caption(f"PDF · max {MAX_UPLOAD_MB} MB with shared key · unlimited with own key")
        uploaded = st.file_uploader("Add a PDF annual report", type="pdf", accept_multiple_files=True)
    if uploaded:
        using_own_key = bool(user_key.strip())
        for f in uploaded:
            if not using_own_key and f.size > MAX_UPLOAD_BYTES:
                st.error(
                    f"**{f.name}** is {f.size / 1024 / 1024:.1f} MB — exceeds the {MAX_UPLOAD_MB} MB limit "
                    "for the shared API key. Add your own OpenAI key above to upload larger files."
                )
                continue

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name
            dest = os.path.join(BUNDLED_DIR, f.name)
            os.replace(tmp_path, dest)
            if not is_ingested(dest):
                with st.spinner(f"Indexing {f.name}…"):
                    n = ingest(dest)
                st.toast(f"Indexed {f.name} ({n} chunks)", icon="✅")
                st.rerun()

    # ── Remove uploaded (non-bundled) docs ────────────────────────────────────
    bundled_basenames = {
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(BUNDLED_DIR, "apple-*.pdf"))
        + glob.glob(os.path.join(BUNDLED_DIR, "hsbc-*.pdf"))
    }
    user_docs = [d for d in docs if d["doc_name"] not in bundled_basenames]
    if user_docs:
        with st.expander("🗑️ Remove uploaded reports"):
            for doc in user_docs:
                if st.button(f"Remove {doc['doc_name']}", key=f"del_{doc['doc_name']}"):
                    delete_doc(doc["doc_name"])
                    pdf_path = os.path.join(BUNDLED_DIR, doc["doc_name"] + ".pdf")
                    if os.path.exists(pdf_path):
                        os.remove(pdf_path)
                    st.rerun()

    st.caption("GPT-4o-mini · ChromaDB · pdfplumber")

# ── Auto-ingest bundled reports on startup ───────────────────────────────────
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
        progress.progress(i / total, text=f"Step {i+1}/{total} — Parsing and embedding **{name}**…")
        n = ingest(pdf)
        progress.progress((i + 1) / total, text=f"✅ {name} indexed ({n} pages)")

    progress.empty()
    st.rerun()

_ingest_bundled()

# ── Main area ─────────────────────────────────────────────────────────────────
st.header("Ask a question")

if not docs:
    st.info("No documents indexed yet. Upload a PDF in the sidebar to get started.")
    st.stop()

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []  # list of {role, content, sources?}

# Clear conversation button
if st.session_state.history:
    if st.button("Clear conversation", icon="🗑️"):
        st.session_state.history = []
        st.rerun()

# ── Display chat history ───────────────────────────────────────────────────────
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            for src in msg["sources"]:
                with st.expander(f"📄 {src['doc_name']}  —  page {src['page_num']}"):
                    st.caption(src["excerpt"])

# ── Chat input ────────────────────────────────────────────────────────────────
question = st.chat_input("Ask a question about the reports…")

if question:
    # Show user message immediately
    with st.chat_message("user"):
        st.write(question)

    # Retrieve and answer
    with st.chat_message("assistant"):
        with st.spinner("Searching and generating answer…"):
            chunks = retrieve(question, k=5)
            result = answer(question, chunks, history=st.session_state.history)
        st.write(result["answer"])
        if result["sources"]:
            for src in result["sources"]:
                with st.expander(f"📄 {src['doc_name']}  —  page {src['page_num']}"):
                    st.caption(src["excerpt"])

    # Save to history
    st.session_state.history.append({"role": "user", "content": question})
    st.session_state.history.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })
