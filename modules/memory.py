"""
modules/memory.py
Handles ChromaDB connection, memory persistence, and adaptive decay.
"""

import chromadb
import os
import time
import sys
from datetime import datetime
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from nanoid import generate


CHROMA_PATH = "/Users/maxwell/PycharmProjects/PythonProject/Context_Augmented_Memory/chroma_db"
CHROMA_PORT = 8000

# --- Ensure Chroma is running ---
def ensure_chroma_running():
    """Ensure Chroma server is running, or prompt the user to start it."""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(("localhost", CHROMA_PORT))
    sock.close()

    if result != 0:
        raise RuntimeError(
            f"❌ Could not reach Chroma.\n➡️  Please start it manually:\n   chroma run --path {CHROMA_PATH} --port {CHROMA_PORT}"
        )
    else:
        print(f"✅ Connected to Chroma on port {CHROMA_PORT}")

# --- Initialize Chroma client and collection ---
def init_memory():
    ensure_chroma_running()
    client = chromadb.HttpClient(host="localhost", port=CHROMA_PORT)
    embedder = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    collection = client.get_or_create_collection(
        name="CAM_Memory",
        embedding_function=embedder,
    )
    print(f"📚 Memory initialized ({collection.count()} records).")
    return collection

collection = init_memory()


# ============================================================
# 🧠 Adaptive Memory Reinforcement System
# ============================================================

def add_memory(user_prompt: str, llm_output: str, tag: str = "NONE"):
    """Add a new memory episode with default neutral reward and full decay weight."""
    episode_id = generate(size=12)
    metadata = {
        "timestamp": datetime.utcnow().isoformat(),
        "user_prompt": user_prompt,
        "tag": tag,
        "reward": 0.0,  # neutral baseline
        "decay": 1.0,   # full relevance
    }

    collection.add(
        ids=[episode_id],
        documents=[llm_output],
        metadatas=[metadata],
    )

    print(f"✅ Added episode {episode_id} (tag: {tag})")
    return episode_id


def update_reward(episode_id: str, reward_delta: float):
    """Increase or decrease reward for a given memory episode."""
    existing = collection.get(ids=[episode_id], include=["metadatas"])
    if not existing["metadatas"][0]:
        print(f"⚠️ No metadata found for episode {episode_id}")
        return

    meta = existing["metadatas"][0][0]
    new_reward = float(meta.get("reward", 0.0)) + reward_delta
    meta["reward"] = max(-1.0, min(1.0, new_reward))  # clamp between -1 and +1

    collection.update(ids=[episode_id], metadatas=[meta])
    print(f"🔁 Updated reward for {episode_id}: {meta['reward']}")


def apply_decay(decay_rate: float = 0.99, min_decay: float = 0.3):
    """
    Gradually decay all low-reward memories.
    Does NOT delete anything — only weakens retrieval weighting.
    """
    data = collection.get(include=["metadatas"])
    if not data or not data.get("metadatas") or not data["metadatas"][0]:
        return

    for i, meta in enumerate(data["metadatas"][0]):
        reward = meta.get("reward", 0.0)
        decay = meta.get("decay", 1.0)

        if reward < 0:
            new_decay = max(min_decay, decay * decay_rate)
            meta["decay"] = new_decay
            collection.update(ids=[data["ids"][0][i]], metadatas=[meta])
            print(f"⬇️ Decayed episode {data['ids'][0][i]} → {new_decay:.3f}")

    print("🧩 Decay cycle complete — no memories deleted.")


def reinforce_memory(episode_id: str):
    """Reinforce a memory when reactivated (distance < threshold)."""
    existing = collection.get(ids=[episode_id], include=["metadatas"])
    if not existing["metadatas"][0]:
        return

    meta = existing["metadatas"][0][0]
    meta["decay"] = min(1.0, meta.get("decay", 1.0) * 1.05)
    collection.update(ids=[episode_id], metadatas=[meta])
    print(f"💪 Reinforced episode {episode_id} → decay {meta['decay']:.3f}")

# --- New helper: Retrieve recent embeddings for context comparison ---
def get_recent_embeddings(n: int = 3):
    """
    Returns embeddings of the most recent n memory entries.
    Used by context_decider to determine semantic continuity.
    """
    try:
        data = collection.get(limit=n, include=["embeddings"])
        if not data or not data.get("embeddings"):
            return []
        return data["embeddings"]
    except Exception as e:
        print(f"⚠️ Failed to get recent embeddings: {e}")
        return []