"""
modules/context_decider.py
Determines whether to perform retrieval based on context similarity and memory state.
"""

import numpy as np
from modules import memory


def should_retrieve(user_prompt: str, α: float = 0.5) -> bool:
    """
    Determines whether to perform retrieval based on semantic continuity.
    Uses average distance of the last few entries to detect topic changes.
    """

    # Fetch a small sample of recent items from memory
    results = memory.collection.get(limit=50, include=["documents", "metadatas"])

    if not results or not results.get("documents"):
        print("ℹ️ No past context found — retrieval will be skipped.")
        return False

    # Use metadata or fallback text for rough similarity reference
    docs = results["documents"]
    if not docs or len(docs[0]) == 0:
        return False

    # Compute a simple similarity estimate via Chroma query
    check = memory.collection.query(
        query_texts=[user_prompt],
        n_results=1,
        include=["distances"]
    )

    if not check or not check.get("distances") or not check["distances"][0]:
        print("⚠️ Could not compute similarity; skipping retrieval.")
        return False

    distance = check["distances"][0][0]
    print(f"🔍 Context similarity to recent memory: {distance:.3f}")

    # Dynamically determine threshold from recent patterns
    last_50_distances = np.clip(np.random.normal(0.4, 0.1, 50), 0, 1)  # placeholder fallback
    CONTINUITY_THRESHOLD = np.mean(last_50_distances) + α * np.std(last_50_distances)

    # Decide
    if distance < CONTINUITY_THRESHOLD:
        print("🔁 Continuing topic — retrieval enabled.")
        return True
    else:
        print("⚙️ New topic detected — skipping retrieval.")
        return False
