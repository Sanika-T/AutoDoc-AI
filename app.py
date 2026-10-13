import os
import time
import streamlit as st
from dotenv import load_dotenv
from rag_engine import (
    build_index, answer_question, clear_index, get_index_stats,
    extract_documents, MODEL_NAME, EMBEDDING_MODEL
)

load_dotenv()

st.set_page_config(
    page_title="AutoDoc AI | HLD Intelligence",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root { --accent: #67e8f9; }
.block-container { padding-top: 1.3rem; padding-bottom: 2.2rem; max-width: 1400px; }
[data-testid="stAppViewContainer"] { background:
  radial-gradient(ellipse at 12% 0%, rgba(20, 184, 166, .12), transparent 34%),
  radial-gradient(ellipse at 95% 8%, rgba(59, 130, 246, .13), transparent 30%); }
.hero {
  padding: 1.5rem 1.7rem; border-radius: 20px;
  border: 1px solid rgba(103,232,249,.28);
  background: linear-gradient(120deg, rgba(15, 23, 42, .96), rgba(17, 55, 73, .92));
  color: white; margin-bottom: 1.1rem;
}
.hero h1 { font-size: 2.1rem; margin: 0; letter-spacing: -.04em; }
.hero p { color: #cbd5e1; margin: .45rem 0 0 0; font-size: 1rem; }
.pill { display:inline-block; padding:.25rem .65rem; border-radius:999px;
  background:rgba(103,232,249,.12); color:#a5f3fc; border:1px solid rgba(103,232,249,.25);
  font-size:.78rem; margin-bottom:.65rem; }
.source-card {
  padding: .85rem 1rem; border-radius: 12px; border: 1px solid rgba(148,163,184,.25);
  background: rgba(100,116,139,.07); margin: .45rem 0;
}
.small-muted { color: #94a3b8; font-size: .85rem; }
.metric-card { border: 1px solid rgba(148,163,184,.22); border-radius: 14px; padding: 1rem; }
div[data-testid="stChatMessage"] { border: 1px solid rgba(148,163,184,.15); border-radius: 14px; }
</style>
""", unsafe_allow_html=True)

if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "chat" not in st.session_state:
    st.session_state.chat = []
if "index_info" not in st.session_state:
    st.session_state.index_info = {}

with st.sidebar:
    st.markdown("## 🚘 AutoDoc AI")
    st.caption("Automotive engineering knowledge assistant")
    st.divider()
    st.markdown("### 1 · Add knowledge")
    uploads = st.file_uploader(
        "Upload HLD / engineering documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Use approved or synthetic documents. Scanned PDFs may need OCR before upload."
    )
    chunk_size = st.slider("Chunk size (characters)", 500, 1800, 1000, 100)
    overlap = st.slider("Chunk overlap", 50, 300, 150, 25)
    top_k = st.slider("Evidence sections to retrieve", 2, 6, 4)
    index_button = st.button("⚡ Build knowledge index", type="primary", use_container_width=True)
    if st.button("Reset current workspace", use_container_width=True):
        clear_index()
        st.session_state.indexed = False
        st.session_state.chat = []
        st.session_state.index_info = {}
        st.rerun()
    st.divider()
    st.markdown("### Model configuration")
    st.caption(f"Answer model: `{MODEL_NAME}`")
    st.caption(f"Embedding model: `{EMBEDDING_MODEL}`")
    st.caption("Gemini API is an external dependency. Do not upload confidential company documents without authorization.")

st.markdown("""
<div class="hero">
  <span class="pill">RAG · SOURCE-GROUNDED ANSWERS · ENGINEERING KNOWLEDGE</span>
  <h1>AutoDoc AI <span style="color:#67e8f9">/</span> HLD Intelligence</h1>
  <p>Ask questions about automotive High-Level Design documents. Explore component responsibilities,
  interfaces and dependencies—with evidence you can verify.</p>
</div>
""", unsafe_allow_html=True)

if not os.getenv("GEMINI_API_KEY"):
    st.warning("Gemini API key not found. Add GEMINI_API_KEY to your `.env` file, then restart the app.")

if index_button:
    if not uploads:
        st.error("Upload at least one PDF, TXT or Markdown document first.")
    elif not os.getenv("GEMINI_API_KEY"):
        st.error("Set GEMINI_API_KEY in your .env file before building the index.")
    else:
        try:
            with st.spinner("Extracting document text and creating semantic embeddings…"):
                docs = extract_documents(uploads)
                info = build_index(docs, chunk_size=chunk_size, overlap=overlap)
            st.session_state.indexed = True
            st.session_state.index_info = info
            st.session_state.chat = []
            st.success(f"Knowledge index ready: {info['chunks']} chunks from {info['documents']} document(s).")
        except Exception as exc:
            st.error(f"Indexing failed: {exc}")
            st.info("Check the API key, internet connection, package installation and API quota.")

indexed = st.session_state.indexed
info = st.session_state.index_info

c1, c2, c3, c4 = st.columns(4)
c1.metric("Workspace", "Ready" if indexed else "Not indexed")
c2.metric("Documents", info.get("documents", 0))
c3.metric("Evidence chunks", info.get("chunks", 0))
c4.metric("Retrieval", "Semantic search" if indexed else "Awaiting data")

tab_chat, tab_guide, tab_about = st.tabs(["💬 Ask AutoDoc", "🧭 Demo guide", "⚙️ How it works"])

with tab_chat:
    left, right = st.columns([1.7, 1])
    with left:
        st.subheader("Engineering Q&A")
        st.caption("Answers are generated from retrieved document evidence. Always verify source passages.")
        if not indexed:
            st.info("Upload your approved or synthetic HLD document in the sidebar and select **Build knowledge index**.")
            st.markdown("**Try these questions after indexing:**")
            st.markdown("- What are the main software components?\n- Which component handles diagnostic notifications?\n- What interfaces connect the assessment and notification components?\n- What is the maximum vehicle speed?")
        for item in st.session_state.chat:
            with st.chat_message(item["role"]):
                st.markdown(item["content"])
                if item.get("sources"):
                    with st.expander(f"🔎 Inspect evidence ({len(item['sources'])} sections)"):
                        for i, src in enumerate(item["sources"], 1):
                            st.markdown(
                                f"<div class='source-card'><b>[{i}] {src['source']} · Page {src['page']}</b>"
                                f"<br><span class='small-muted'>Retrieval similarity: {src['similarity']:.3f}</span>"
                                f"<br>{src['text']}</div>", unsafe_allow_html=True
                            )
        prompt = st.chat_input("Ask about components, interfaces, data flow or requirements…", disabled=not indexed)
        if prompt and indexed:
            st.session_state.chat.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Finding evidence and composing a grounded answer…"):
                    try:
                        result = answer_question(prompt, top_k=top_k)
                        answer = result["answer"]
                        st.markdown(answer)
                        st.session_state.chat.append({
                            "role": "assistant", "content": answer, "sources": result["sources"]
                        })
                        if result["sources"]:
                            with st.expander(f"🔎 Inspect evidence ({len(result['sources'])} sections)", expanded=True):
                                for i, src in enumerate(result["sources"], 1):
                                    st.markdown(
                                        f"<div class='source-card'><b>[{i}] {src['source']} · Page {src['page']}</b>"
                                        f"<br><span class='small-muted'>Retrieval similarity: {src['similarity']:.3f}</span>"
                                        f"<br>{src['text']}</div>", unsafe_allow_html=True
                                    )
                    except Exception as exc:
                        st.error(f"Question failed: {exc}")
                        st.info("Check the API key, quota and internet connection.")
            st.rerun()
    with right:
        st.subheader("Workspace insights")
        if indexed:
            st.success("Your document index is active.")
            st.markdown("**Indexed files**")
            for name in info.get("file_names", []):
                st.markdown(f"- `{name}`")
            st.markdown("**Trust controls**")
            st.markdown("- Source page shown for each evidence chunk\n- Answers restricted to retrieved context\n- Explicit fallback when evidence is missing\n- Human verification encouraged")
        else:
            st.markdown("""
            <div class="source-card">
            <b>Designed for engineering traceability</b><br><br>
            <span class="small-muted">AutoDoc AI retrieves relevant sections before asking Gemini to answer.
            The evidence panel lets reviewers check the source instead of trusting an unsupported response.</span>
            </div>
            """, unsafe_allow_html=True)

with tab_guide:
    st.subheader("Suggested 5-minute demonstration flow")
    steps = [
        ("01", "Load the knowledge base", "Upload the included synthetic sample HLD or an approved project document, then build the index."),
        ("02", "Ask a supported question", "Ask about components, interface names, responsibilities or a described data flow."),
        ("03", "Inspect evidence", "Open the evidence panel and verify the source file, page number and retrieved passage."),
        ("04", "Test the boundary", "Ask for a value that is absent from the document. The assistant should say it cannot find supporting evidence."),
        ("05", "Explain evaluation", "Show the prepared evaluation_questions.csv, answer examples, source-grounding checks and limitations.")
    ]
    for number, title, desc in steps:
        st.markdown(f"### {number} · {title}\n{desc}")
    st.warning("Use only approved or synthetic input data. The included sample document is synthetic and is not an official AUTOSAR specification.")

with tab_about:
    st.subheader("System architecture")
    st.code("""
User → Streamlit UI → PDF/TXT extraction
     → Text chunking with overlap
     → Gemini embedding model
     → ChromaDB vector store
     → Similarity retrieval (top-k)
     → Gemini Flash answer generation
     → Answer + source citations + evidence inspection
""", language="text")
    st.markdown("**Key design choices**")
    st.markdown("""
- **RAG:** retrieves relevant evidence before generating an answer.
- **Semantic retrieval:** uses embeddings rather than only exact keyword matches.
- **Grounding instruction:** asks the model not to invent information missing from the context.
- **Source traceability:** each chunk retains its source filename and page number.
- **Evaluation readiness:** a small ground-truth question set and an output CSV template are included.
""")
    st.markdown("**Known limitations**")
    st.markdown("""
- Scanned PDFs may require OCR before ingestion.
- Retrieval similarity is a ranking signal, not a calibrated confidence score.
- Gemini API availability, quota and model names can change.
- This prototype is not a replacement for formal engineering review or safety approval.
""")

st.markdown("---")
st.caption("AutoDoc AI · Academic prototype · Gemini API usage and document sharing must follow applicable policies.")
