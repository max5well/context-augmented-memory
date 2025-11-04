"""
modules/memory.py
Handles storing and accessing memory in Chroma.
"""

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from nanoid import generate
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Initialize Chroma client (persistent local DB) ---
client = chromadb.PersistentClient(path="./chroma_db")

# --- Embedding function ---
embedder = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# --- Shared Chroma collection ---
collection = client.get_or_create_collection(
    name="CAM_collection",
    embedding_function=embedder,
)


def add_episode(prompt: str, llm_output: str, metadata: dict) -> str:
    """
    Store an interaction (prompt + output + metadata) into Chroma.
    """
    episode_id = generate(size=12)

    # Each entry stores the model output as the document
    collection.add(
        ids=[episode_id],
        documents=[promp],
        metadatas=[metadata],
    )

    print(f"✅ Added episode {episode_id}")
    return episode_id
