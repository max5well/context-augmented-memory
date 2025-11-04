"""
modules/retrieval.py
Handles retrieval of relevant context from Chroma memory.
"""

import modules.memory as memory
from typing import List, Tuple, Dict

# Configurable distance threshold
MAX_DISTANCE = 0.6  # more forgiving to ensure matches are found


def retrieve_context(query: str, n_results: int = 5) -> str:
    """
    Retrieve relevant past memory entries for a given query.
    Filters by distance threshold and returns a formatted context string.
    """
    results = memory.collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "distances", "metadatas"],
    )

    # Debug output to verify raw query results
    print("\n🔍 Raw retrieval results:")
    print(results)

    if not results or not results.get("documents") or not results["documents"][0]:
        print("⚠️ No matching memory found (empty results).")
        return ""

    docs = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    # Combine and filter by distance
    relevant: List[Tuple[str, float, Dict]] = [
        (doc, dist, meta)
        for doc, dist, meta in zip(docs, distances, metadatas)
        if dist <= MAX_DISTANCE
    ]

    # If nothing under threshold, take top result for debugging visibility
    if not relevant and docs:
        print(f"⚠️ No results under threshold ({MAX_DISTANCE}), returning closest match.")
        relevant = [(docs[0], distances[0], metadatas[0])]

    # Format retrieved content
    context_lines = [
        f"[Memory — tag: {meta.get('tag', 'unknown')} | distance: {dist:.3f}]\n{doc}\n"
        for doc, dist, meta in relevant
    ]

    print(f"✅ Retrieved {len(relevant)} relevant memories (distance ≤ {MAX_DISTANCE})")

    # Join context into one text block
    context_text = "\n---\n".join(context_lines)

    # Return for use in main.py
    return context_text
