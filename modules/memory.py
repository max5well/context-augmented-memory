"""
memory.py
Handles Chroma vector storage and retrieval for CAM.
Clean version — uses OpenAI embeddings only (no Chroma auto-embedding).
"""

import os
import chromadb
from chromadb.config import Settings
from modules import embedding

# --- Initialize Chroma client ---
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = os.getenv("CHROMA_PORT", "8001")

# Try to connect to Chroma server, fall back to embedded if server not available
try:
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    print(f"🌐 Connected to Chroma server at {CHROMA_HOST}:{CHROMA_PORT}")
except Exception as e:
    print(f"⚠️ Chroma server not available, using embedded client: {e}")
    CHROMA_PATH = os.path.abspath("./CAM_project/chroma_db")
    client = chromadb.Client(Settings(
        persist_directory=CHROMA_PATH,
        anonymized_telemetry=False
    ))

COLLECTION_NAME = "cam_memory"

def get_embedding_dimension() -> int:
    """Return current embedding model's vector dimension."""
    sample = embedding.get_embedding("dimension check")
    dim = len(sample) if sample else 0
    print(f"🧮 Detected embedding dimension: {dim}")
    return dim

# ✅ Disable Chroma's built-in embedding model since we use OpenAI embeddings
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=None
)

print(f"🔍 Using collection: {COLLECTION_NAME}")
print(f"📊 Current entries: {collection.count()}")

# --- Core memory functions ---

def store(text: str, metadata: dict, embedding_vector: list):
    """Store new memory record in Chroma."""
    if not embedding_vector:
        print("⚠️ Skipping storage — empty embedding vector.")
        return

    try:
        id_ = metadata.get("episode_id", "unknown")
        print(f"📝 Storing memory {id_} with {len(embedding_vector)} dims")

        collection.add(
            documents=[text],
            metadatas=[metadata],
            embeddings=[embedding_vector],
            ids=[id_],
        )

        # Verify it was stored
        count = collection.count()
        print(f"✅ Memory stored. Total entries: {count}")

    except Exception as e:
        print(f"❌ Failed to store memory: {e}")
        import traceback
        traceback.print_exc()

def query(query_text: str, n_results: int = 5):
    """Query similar memories."""
    return collection.query(query_texts=[query_text], n_results=n_results)

def get_recent_embeddings(n: int = 3):
    """
    Returns the most recent n embeddings from memory for context comparison.
    """
    try:
        data = collection.peek(n)
        embeddings = data.get("embeddings", [])

        if embeddings is None or len(embeddings) == 0:
            print("⚠️ No recent embeddings found.")
            return []

        # Some Chroma versions return NumPy arrays, others lists — normalize them
        normalized = []
        for e in embeddings:
            if e is None:
                continue
            import numpy as np
            if isinstance(e, np.ndarray):
                normalized.append(e.tolist())
            elif isinstance(e, list):
                normalized.append(e)
            else:
                try:
                    normalized.append(list(e))
                except Exception:
                    continue

        print(f"📤 Retrieved {len(normalized)} recent embeddings for comparison.")
        return normalized

    except Exception as e:
        print(f"⚠️ Failed to get recent embeddings: {e}")
        return []

