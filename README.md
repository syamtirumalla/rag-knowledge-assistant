# 🧠 RAG Knowledge Assistant

An AI-powered document Q&A system built with Retrieval-Augmented Generation (RAG). Upload any PDF and ask questions — get grounded answers with source citations and a 3D knowledge map.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.14-green) ![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)

## ✨ Features
- 📄 Upload and index PDF documents
- 💬 Chat interface with context-aware answers
- 📎 Source citations with confidence scores (HIGH/MEDIUM/LOW)
- 🗺️ Interactive 3D Knowledge Map using UMAP + Plotly
- 🚫 Hallucination prevention via confidence-score gating

## 🛠️ Tech Stack
| Layer | Technology |
|-------|-----------|
| Framework | LlamaIndex |
| LLM | Groq (Llama 3.3 70B) |
| Embeddings | BAAI/bge-large-en-v1.5 |
| Vector DB | ChromaDB |
| Visualization | UMAP + Plotly |
| UI | Streamlit |

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/syamtirumalla/rag-knowledge-assistant.git
cd rag-knowledge-assistant

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Add your Groq API key
echo "GROQ_API_KEY=your_key_here" > .env

# Run the app
streamlit run app.py
```

## 📁 Project Structure

rag-knowledge-assistant/
├── app.py              # Streamlit UI
├── ingestor.py         # PDF ingestion & ChromaDB indexing
├── retriever.py        # Semantic search & source retrieval
├── llm.py              # Groq LLM setup
├── rag_pipeline.py     # End-to-end RAG pipeline
├── visualizer.py       # UMAP 3D knowledge map
└── requirements.txt

## 🔑 Get a Free Groq API Key
Visit [console.groq.com](https://console.groq.com) to get a free API key.

## 👥 Built By
- K L S K Vinay
- Tirumalla Syam Narayana  
- Chandragiri Venkat Mourya
- Allu Atchiyya Naidu

NIT Silchar — Summer Internship 2024