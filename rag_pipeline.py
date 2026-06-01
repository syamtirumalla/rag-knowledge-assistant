import os
from dotenv import load_dotenv
from llama_index.llms.groq import Groq
from llama_index.core import Settings
from ingestor import load_existing_index, get_doc_count
from retriever import query_with_sources

load_dotenv()

def setup_llm():
    api_key = os.getenv("GROQ_API_KEY")
    llm = Groq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.1,
        max_tokens=1024,
    )
    Settings.llm = llm
    return llm

def ask(question: str):
    print(f"\nQuestion: {question}")
    print("-" * 50)

    # Load LLM
    setup_llm()

    # Load index
    index = load_existing_index()
    if not index:
        print("No documents indexed yet! Run ingestor.py first.")
        return

    # Query with sources
    answer, sources = query_with_sources(index, question)

    if not answer:
        print("No relevant information found in the knowledge base.")
        return

    print(f"\nAnswer:\n{answer}")
    print(f"\nSources found: {len(sources)}")
    for i, src in enumerate(sources, 1):
        print(f"\n[Source {i}] File: {src['file']} | Page: {src['page']}")
        print(f"  Confidence: {src['confidence'].upper()} (score: {src['score']})")
        print(f"  Text: {src['text'][:200]}...")

if __name__ == "__main__":
    print(f"Documents in knowledge base: {get_doc_count()}")
    
    # Test questions
    questions = [
        "What is RAG and how does it work?",
        "What are the best vector databases?",
        "What is the difference between PCA and UMAP?",
    ]
    
    for q in questions:
        ask(q)
        print("\n" + "="*60)