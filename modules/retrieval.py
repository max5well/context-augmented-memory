"""
retrieval.py
Handles retrieval of relevant context from Chroma memory.
Optimized for:
- lightweight pronoun-aware re-ranking
- adaptive distance threshold logic
- flexible dual retrieval modes ("contextual" and "global")
"""

from modules import memory, embedding, config_manager
from typing import List, Tuple, Dict
import numpy as np

# Load config values
config = config_manager.load_config()
BASE_DISTANCE = config["retrieval"]["max_distance"]


def _rerank_with_pronouns(
    query: str,
    results: List[Tuple[str, float, Dict]],
) -> List[Tuple[str, float, Dict]]:
    """
    Lightweight re-ranking of retrieval results.
    Prioritizes memories containing named entities that match pronoun references.
    Example:
        query = "How old is he?" → boost entries with "Tom" or other recent entities.
    """
    pronouns = {"he", "she", "it", "they", "him", "her", "them"}
    if not any(p in query.lower().split() for p in pronouns):
        return results  # No pronouns → skip

    # Simple heuristic boost: entries with capitalized names or "called/named"
    reranked = []
    for doc, dist, meta in results:
        score = 1.0 / (1.0 + dist)  # baseline inverse distance weight
        text = (meta.get("user_prompt", "") + " " + doc).lower()
        if any(kw in text for kw in ["called", "named"]):
            score *= 1.25
        if any(word.istitle() for word in doc.split()):  # capitalized names
            score *= 1.1
        reranked.append((doc, dist, meta, score))

    # Sort descending by score
    reranked.sort(key=lambda x: x[3], reverse=True)
    return [(doc, dist, meta) for doc, dist, meta, _ in reranked]


def retrieve_context(
    query: str,
    n_results: int = 5,
    include_meta: bool = False,
    mode: str = "contextual",
) -> str:
    """
    Retrieves relevant memory entries for a given query.

    mode:
        - "contextual" → limited to facts & continuity checks
        - "global" → full memory search across all intents
    """

    where_filter = {"intent": "fact"} if mode == "contextual" else {}

    # --- Generate query embedding manually ---
    query_vector = embedding.get_embedding(query)
    if not query_vector:
        print("⚠️ Failed to generate query embedding.")
        return ""

    # --- Perform Chroma query ---
    results = memory.collection.query(
        query_embeddings=[query_vector],
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

    # --- Adaptive filtering logic ---
    threshold = BASE_DISTANCE
    relevant: List[Tuple[str, float, Dict]] = [
        (doc, dist, meta)
        for doc, dist, meta in zip(docs, distances, metadatas)
        if dist <= threshold
    ]

    # If nothing passes filter, relax dynamically
    if not relevant and distances:
        avg_dist = np.mean(distances)
        new_threshold = min(avg_dist + 0.25, 1.2)
        print(f"⚙️ Relaxing threshold {threshold:.2f} → {new_threshold:.2f} (avg_dist={avg_dist:.3f})")

        relevant = [
            (doc, dist, meta)
            for doc, dist, meta in zip(docs, distances, metadatas)
            if dist <= new_threshold
        ]
        threshold = new_threshold

    if not relevant:
        print(f"⚠️ No relevant items under distance threshold ({threshold:.2f}).")
        return ""

    # --- Pronoun-aware re-ranking ---
    relevant = _rerank_with_pronouns(query, relevant)

    # --- Build readable context ---
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

    print(f"✅ Retrieved {len(relevant)} relevant memories (mode={mode}, distance ≤ {threshold:.2f})")
    return "\n---\n".join(context_lines)
