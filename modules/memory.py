"""
memory.py
Handles Chroma vector storage and retrieval for CAM.
Now includes automatic embedding dimension detection and recovery.
"""

import os
import chromadb
from chromadb.config import Settings
from modules import embedding

# --- Initialize Chroma client ---
CHROMA_PATH = os.path.abspath("./CAM_project/chroma_db")
client = chromadb.Client(Settings(
    persist_directory=CHROMA_PATH,
    anonymized_telemetry=False
))

COLLECTION_NAME = "cam_memory"

def get_embedding_dimension() -> int:
    """Return current embedding model's vector dimension."""
    sample = embedding.get_embedding("dimension check")
    return len(sample) if sample else 0

def ensure_collection_consistency():
    """Ensure Chroma collection matches embedding model dimension."""
    dim = get_embedding_dimension()
    print(f"🧮 Detected embedding dimension: {dim}")

    try:
        collection = client.get_collection(COLLECTION_NAME)
        # Check an existing record to compare
        sample = collection.peek(1)
        if sample and "embeddings" in sample and len(sample["embeddings"][0]) != dim:
            print(f"⚠️ Dimension mismatch detected — resetting collection ({len(sample['embeddings'][0])} → {dim})")
            client.delete_collection(COLLECTION_NAME)
            return client.create_collection(COLLECTION_NAME)
        return collection
    except Exception:
        print("ℹ️ No existing collection found — creating new.")
        return client.create_collection(COLLECTION_NAME)

# --- Initialize collection ---
collection = ensure_collection_consistency()

# --- Core memory functions ---

def store(text: str, metadata: dict, embedding_vector: list):
    """Store new memory record in Chroma."""
    if not embedding_vector:
        print("⚠️ Skipping storage — empty embedding vector.")
        return

    try:
        id_ = metadata.get("episode_id", "unknown")
        collection.add(
            documents=[text],
            metadatas=[metadata],
            embeddings=[embedding_vector],
            ids=[id_],
        )
        print(f"🧠 Stored memory: {id_} ({len(embedding_vector)} dims)")
    except Exception as e:
        print(f"❌ Failed to store memory: {e}")

def query(query_text: str, n_results: int = 5):
    """Query similar memories."""
    return collection.query(query_texts=[query_text], n_results=n_results)


def get_recent_embeddings(n: int = 3):
    """
    Returns the most recent n embeddings from memory for context comparison.
    """
    try:
        data = collection.peek(n)
        if not data or "embeddings" not in data or not data["embeddings"]:
            print("⚠️ No recent embeddings found.")
            return []

        embeddings = [e for e in data["embeddings"] if e]
        print(f"📤 Retrieved {len(embeddings)} recent embeddings for comparison.")
        return embeddings
    except Exception as e:
        print(f"⚠️ Failed to get recent embeddings: {e}")
        return []
