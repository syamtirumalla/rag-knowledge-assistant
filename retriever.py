from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import Settings
import re

def get_query_engine(index: VectorStoreIndex, top_k: int = 5, threshold: float = 0.3):
    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
    postprocessor = SimilarityPostprocessor(similarity_cutoff=threshold)
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=[postprocessor],
    )
    return query_engine

def query_with_sources(index, question: str, top_k: int = 5):
    from llama_index.core.prompts import PromptTemplate
    from llama_index.core import Settings
    import groq as groq_sdk
    import os

    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
    nodes = retriever.retrieve(question)

    if not nodes:
        return None, []

    sources = []
    context_parts = []
    for node in nodes:
        score = node.score if node.score else 0.0
        if score >= 0.85:
            confidence = "high"
            color = "green"
        elif score >= 0.50:
            confidence = "medium"
            color = "orange"
        else:
            confidence = "low"
            color = "red"

        sources.append({
            "text": node.node.text,
            "score": round(score, 3),
            "confidence": confidence,
            "color": color,
            "file": node.node.metadata.get("file_name", "Unknown"),
            "page": node.node.metadata.get("page_label", "?"),
        })
        context_parts.append(node.node.text)

    # Build context and call Groq directly
    context = "\n\n".join(context_parts[:3])  # top 3 chunks only
    
    client = groq_sdk.Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Answer ONLY using the provided context. Be concise — 3 to 5 sentences max. If the answer is not in the context, say 'I don't have enough information on that.'"
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ],
        max_tokens=512,
        temperature=0.1,
    )

    answer = response.choices[0].message.content
    return answer, sources