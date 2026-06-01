import numpy as np
import plotly.graph_objects as go
from umap import UMAP
from ingestor import get_chroma_collection

def get_embeddings_from_chroma():
    _, collection = get_chroma_collection()
    result = collection.get(include=["embeddings", "documents", "metadatas"])
    
    embeddings = np.array(result["embeddings"])
    documents = result["documents"]
    metadatas = result["metadatas"]
    return embeddings, documents, metadatas

def build_3d_knowledge_map():
    embeddings, documents, metadatas = get_embeddings_from_chroma()
    
    if len(embeddings) < 4:
        return None, "Need at least 4 chunks to build visualization."

    print(f"Reducing {len(embeddings)} embeddings to 3D with UMAP...")
    reducer = UMAP(n_components=3, n_neighbors=min(15, len(embeddings)-1), min_dist=0.1, random_state=42)
    coords = reducer.fit_transform(embeddings)

    # Build hover text
    hover_texts = []
    colors = []
    for i, (doc, meta) in enumerate(zip(documents, metadatas)):
        file = meta.get("file_name", "Unknown") if meta else "Unknown"
        page = meta.get("page_label", "?") if meta else "?"
        preview = doc[:120].replace("<", "&lt;").replace(">", "&gt;")
        hover_texts.append(f"<b>File:</b> {file}<br><b>Page:</b> {page}<br><b>Text:</b> {preview}...")
        colors.append(i)

    fig = go.Figure(data=[go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode="markers+text",
        marker=dict(
            size=8,
            color=colors,
            colorscale="Viridis",
            opacity=0.85,
            colorbar=dict(title="Chunk Index")
        ),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover_texts,
    )])

    fig.update_layout(
        title="📚 Knowledge Map — 3D Embedding Space",
        scene=dict(
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2",
            zaxis_title="UMAP-3",
            bgcolor="rgb(10,10,30)",
            xaxis=dict(gridcolor="rgb(50,50,80)"),
            yaxis=dict(gridcolor="rgb(50,50,80)"),
            zaxis=dict(gridcolor="rgb(50,50,80)"),
        ),
        paper_bgcolor="rgb(15,15,35)",
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=40, b=0),
        height=700,
    )

    return fig, f"Mapped {len(embeddings)} chunks into 3D space."

if __name__ == "__main__":
    fig, msg = build_3d_knowledge_map()
    print(msg)
    if fig:
        fig.write_html("knowledge_map.html")
        print("Saved! Open knowledge_map.html in your browser.")