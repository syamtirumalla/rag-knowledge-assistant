import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        color: #1a1a2e;
    }
    .hero-sub {
        color: #666;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }
    .source-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 6px;
    }
    .high   { background:#dcfce7; color:#16a34a; }
    .medium { background:#fef9c3; color:#ca8a04; }
    .low    { background:#fee2e2; color:#dc2626; }
</style>
""", unsafe_allow_html=True)

from ingestor import ingest_documents, load_existing_index, get_doc_count
from retriever import query_with_sources
from visualizer import build_3d_knowledge_map
from llm import setup_llm

if "messages" not in st.session_state:
    st.session_state.messages = []
if "index" not in st.session_state:
    st.session_state.index = None
if "llm_ready" not in st.session_state:
    st.session_state.llm_ready = False

if not st.session_state.llm_ready:
    try:
        setup_llm()
        st.session_state.llm_ready = True
    except Exception as e:
        st.error(f"LLM setup failed: {e}")

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=60)
    st.title("RAG Assistant")
    st.caption("Powered by LlamaIndex · Groq · ChromaDB")
    st.divider()

    st.subheader("📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    if uploaded_files:
        for f in uploaded_files:
            with open(os.path.join("data", f.name), "wb") as out:
                out.write(f.getbuffer())
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded")

    if st.button("⚡ Index Documents", use_container_width=True, type="primary"):
        with st.spinner("Indexing..."):
            try:
                import chromadb
                client = chromadb.PersistentClient(path="./chroma_db")
                client.delete_collection("rag_docs")
            except:
                pass
            index, msg = ingest_documents()
            if index:
                st.session_state.index = index
                st.success(msg)
            else:
                st.error(msg)

    st.divider()
    st.subheader("📊 Stats")
    doc_count = get_doc_count()
    col1, col2 = st.columns(2)
    col1.metric("Chunks", doc_count)
    col2.metric("LLM", "🟢 On" if st.session_state.llm_ready else "🔴 Off")

    if st.session_state.index is None and doc_count > 0:
        st.session_state.index = load_existing_index()

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Main ──────────────────────────────────────────────────
st.markdown('<p class="hero-title">🧠 Knowledge Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Ask anything about your documents — grounded answers with cited sources</p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["💬 Chat", "🗺️ Knowledge Map"])

with tab1:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
                with st.expander(f"📎 {len(msg['sources'])} sources used"):
                    for i, src in enumerate(msg["sources"], 1):
                        conf = src["confidence"]
                        badge = f'<span class="source-badge {conf}">{conf.upper()}</span>'
                        st.markdown(
                            f'{badge} **{src["file"]}** · Page {src["page"]} · Score `{src["score"]}`',
                            unsafe_allow_html=True
                        )
                        st.caption(src["text"][:300] + "...")
                        if i < len(msg["sources"]):
                            st.divider()

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if not st.session_state.index:
                answer = "⚠️ No documents indexed yet! Upload PDFs and click **⚡ Index Documents**."
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                with st.spinner("Searching knowledge base..."):
                    answer, sources = query_with_sources(st.session_state.index, prompt)
                if not answer:
                    answer = "I couldn't find relevant information for that question."
                st.markdown(answer)
                if sources:
                    with st.expander(f"📎 {len(sources)} sources used"):
                        for i, src in enumerate(sources, 1):
                            conf = src["confidence"]
                            badge = f'<span class="source-badge {conf}">{conf.upper()}</span>'
                            st.markdown(
                                f'{badge} **{src["file"]}** · Page {src["page"]} · Score `{src["score"]}`',
                                unsafe_allow_html=True
                            )
                            st.caption(src["text"][:300] + "...")
                            if i < len(sources):
                                st.divider()
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

with tab2:
    st.subheader("🗺️ 3D Knowledge Map")
    st.caption("Each dot is a document chunk projected into 3D using UMAP. Hover to preview. Rotate to explore.")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Generate Knowledge Map", use_container_width=True, type="primary"):
            with st.spinner("Building 3D map..."):
                fig, msg = build_3d_knowledge_map()
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    st.success(msg)
                else:
                    st.warning(msg)