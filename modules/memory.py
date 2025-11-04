# modules/retrieval.py
"""
Handles retrieval of relevant context from Chroma memory.
Reuses the shared Chroma collection from memory.py.
"""

from modules.memory import collection
from typing import List, Tuple, Dict

# Configurable distance threshold
MAX_DISTANCE = 0.25  # Lower = stricter; tune per embedding model


def retrieve_context(query: str, n_results: int = 5) -> str:
    """
    Retrieve relevant past memory entries for a given query.
    Filters by distance threshold and returns a context string.
    """
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "distances", "metadatas"],
    )

    if not results or not results.get("documents"):
        print("⚠️ No matching memory found.")
        return ""

    docs = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    # Filter by distance
    relevant: List[Tuple[str, float, Dict]] = [
        (doc, dist, meta)
        for doc, dist, meta in zip(docs, distances, metadatas)
        if dist <= MAX_DISTANCE
    ]

    if not relevant:
        print(f"⚠️ No relevant items under distance threshold ({MAX_DISTANCE}).")
        return ""

    # Build formatted text block
    context_lines = [
        f"[Memory — tag: {meta.get('tag', 'unknown')} | distance: {dist:.3f}]\n{doc}\n"
        for doc, dist, meta in relevant
    ]

    print(f"✅ Retrieved {len(relevant)} relevant memories (distance ≤ {MAX_DISTANCE})")

    return "\n---\n".join(context_lines)
