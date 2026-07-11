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
    .stApp { background-color: #0d0d0d; }
    [data-testid="stSidebar"] {
        background-color: #171717;
        border-right: 1px solid #2a2a2a;
    }
    header[data-testid="stHeader"] { background: transparent; }
    p, li, label, div { color: #ececec; }
    h1, h2, h3 { color: #ffffff; }

    [data-testid="stChatInput"] textarea {
        background: #2a2a2a !important;
        color: #ececec !important;
        border-radius: 16px !important;
        font-size: 15px !important;
    }
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
    }
    .stButton > button {
        background: #2a2a2a;
        color: #ececec;
        border: 1px solid #3a3a3a;
        border-radius: 8px;
    }
    .stButton > button[kind="primary"] {
        background: #10a37f;
        border-color: #10a37f;
        color: white;
    }
    [data-testid="stMetric"] {
        background: #1a1a1a;
        border-radius: 8px;
        border: 1px solid #2a2a2a;
        padding: 8px;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: #1a1a1a;
        border-radius: 8px;
        padding: 3px;
    }
    .stTabs [aria-selected="true"] {
        background: #10a37f !important;
        color: white !important;
        border-radius: 6px;
    }
    .stTabs [data-baseweb="tab"] { color: #8e8ea0 !important; }
    [data-testid="stExpander"] {
        background: #1a1a1a;
        border: 1px solid #2a2a2a !important;
        border-radius: 8px;
    }
    hr { border-color: #2a2a2a !important; }
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: #0d0d0d; }
    ::-webkit-scrollbar-thumb { background: #3a3a3a; border-radius: 3px; }
    .badge-high   { background:#0a2e20; color:#10a37f; border:1px solid #10a37f; padding:2px 10px; border-radius:20px; font-size:11px; font-weight:600; display:inline-block; }
    .badge-medium { background:#2e1f0a; color:#f0a500; border:1px solid #f0a500; padding:2px 10px; border-radius:20px; font-size:11px; font-weight:600; display:inline-block; }
    .badge-low    { background:#2e0a0a; color:#f05050; border:1px solid #f05050; padding:2px 10px; border-radius:20px; font-size:11px; font-weight:600; display:inline-block; }
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

with st.sidebar:
    st.markdown("### 🧠 RAG Assistant")
    st.caption("LlamaIndex · Groq · ChromaDB")
    st.divider()

    st.markdown("**📂 Upload Documents**")
    uploaded_files = st.file_uploader(
        "Upload PDFs", type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    if uploaded_files:
        for f in uploaded_files:
            with open(os.path.join("data", f.name), "wb") as out:
                out.write(f.getbuffer())
        st.success(f"✅ {len(uploaded_files)} file(s) ready")

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
    st.markdown("**📊 Stats**")
    doc_count = get_doc_count()
    col1, col2 = st.columns(2)
    col1.metric("Chunks", doc_count)
    col2.metric("LLM", "🟢" if st.session_state.llm_ready else "🔴")

    if st.session_state.index is None and doc_count > 0:
        st.session_state.index = load_existing_index()

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main
tab1, tab2 = st.tabs(["💬  Chat", "🗺️  Knowledge Map"])

with tab1:
    if not st.session_state.messages:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align:center; color:#ececec'>What's on your mind today?</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color:#8e8ea0'>Upload a PDF and ask anything about it</p>", unsafe_allow_html=True)
        st.markdown("<br><br>", unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
                with st.expander(f"📎 {len(msg['sources'])} sources"):
                    for i, src in enumerate(msg["sources"], 1):
                        conf = src["confidence"]
                        badge = f'<span class="badge-{conf}">{conf.upper()}</span>'
                        st.markdown(f'{badge} &nbsp; **{src["file"]}** · Page {src["page"]} · Score `{src["score"]}`', unsafe_allow_html=True)
                        st.caption(src["text"][:300] + "...")
                        if i < len(msg["sources"]):
                            st.divider()

    if prompt := st.chat_input("Ask anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if not st.session_state.index:
                answer = "⚠️ No documents indexed yet! Upload PDFs and click **⚡ Index Documents**."
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                with st.spinner("Thinking..."):
                    answer, sources = query_with_sources(st.session_state.index, prompt)
                if not answer:
                    answer = "I couldn't find relevant information for that question."
                st.markdown(answer)
                if sources:
                    with st.expander(f"📎 {len(sources)} sources"):
                        for i, src in enumerate(sources, 1):
                            conf = src["confidence"]
                            badge = f'<span class="badge-{conf}">{conf.upper()}</span>'
                            st.markdown(f'{badge} &nbsp; **{src["file"]}** · Page {src["page"]} · Score `{src["score"]}`', unsafe_allow_html=True)
                            st.caption(src["text"][:300] + "...")
                            if i < len(sources):
                                st.divider()
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

with tab2:
    st.markdown("### 🗺️ 3D Knowledge Map")
    st.caption("Each dot is a document chunk in 3D space. Hover to preview. Rotate to explore.")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Generate Knowledge Map", use_container_width=True, type="primary"):
            with st.spinner("Building 3D map with UMAP..."):
                fig, msg = build_3d_knowledge_map()
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    st.success(msg)
                else:
                    st.warning(msg)