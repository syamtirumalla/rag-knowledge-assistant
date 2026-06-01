import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

CHROMA_PATH = "./chroma_db"
DATA_PATH = "./data"
EMBED_MODEL = "BAAI/bge-large-en-v1.5"

def get_embed_model():
    return HuggingFaceEmbedding(model_name=EMBED_MODEL)

def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection("rag_docs")
    return client, collection

def ingest_documents():
    if not os.listdir(DATA_PATH):
        return None, "No documents found in /data folder."

    print("Loading documents...")
    from llama_index.core import SimpleDirectoryReader
    documents = SimpleDirectoryReader(
        DATA_PATH,
        required_exts=[".pdf"],
        recursive=True
    ).load_data()

    print("Chunking documents...")
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    print("Loading embedding model...")
    Settings.embed_model = get_embed_model()
    Settings.llm = None

    print("Setting up ChromaDB...")
    client, collection = get_chroma_collection()
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("Building index...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[splitter],
        show_progress=True
    )

    print(f"Done! Indexed {len(documents)} document(s).")
    return index, f"Successfully indexed {len(documents)} document(s)."

def load_existing_index():
    client, collection = get_chroma_collection()
    if collection.count() == 0:
        return None

    Settings.embed_model = get_embed_model()
    Settings.llm = None

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )
    return index

def get_doc_count():
    _, collection = get_chroma_collection()
    return collection.count()

if __name__ == "__main__":
    index, msg = ingest_documents()
    print(msg)