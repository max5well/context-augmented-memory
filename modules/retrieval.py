"""
retrieval.py
Handles retrieval of relevant context from Chroma memory.
Dynamically configured via config_manager.py.
"""

from modules import memory, config_manager
from typing import List, Tuple, Dict

# Load dynamic configuration
config = config_manager.load_config()
MAX_DISTANCE = config["retrieval"]["max_distance"]


def retrieve_context(query: str, n_results: int = 5, include_meta: bool = False) -> str:
    """
    Retrieves relevant memory entries for a given query.
    Uses configurable distance threshold from config.json.
    If include_meta=True, includes timestamps and metadata.
    """

    where_filter = {} if include_meta else {"intent": "fact"}

    results = memory.collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter,
        include=["documents", "distances", "metadatas"],
    )

    if not results or not results.get("documents") or not results["documents"][0]:
        print("⚠️ No matching memory found.")
        return ""

    docs = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    relevant: List[Tuple[str, float, Dict]] = [
        (doc, dist, meta)
        for doc, dist, meta in zip(docs, distances, metadatas)
        if dist <= MAX_DISTANCE
    ]

    if not relevant:
        print(f"⚠️ No relevant items under distance threshold ({MAX_DISTANCE}).")
        return ""

    # Build output
    if include_meta:
        context_lines = [
            f"[Meta — tag: {meta.get('tag', 'NONE')} | date: {meta.get('timestamp', 'unknown')}]"
            f"\nUser said: {meta.get('user_prompt', 'N/A')}\nStored output: {doc}\n"
            for doc, _, meta in relevant
        ]
    else:
        context_lines = [
            f"[Memory — tag: {meta.get('tag', 'NONE')} | distance: {dist:.3f}]"
            f"\n{doc}\n"
            for doc, dist, meta in relevant
        ]

    print(f"✅ Retrieved {len(relevant)} relevant memories (distance ≤ {MAX_DISTANCE})")
    return "\n---\n".join(context_lines)
