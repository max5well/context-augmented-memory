"""
retrieval.py
Handles retrieval of relevant context from Chroma memory.

Supports:
- contextual vs global search modes
- adaptive distance thresholds
- pronoun-aware re-ranking
- safe Chroma query filters
- plain (fact-authoritative) retrieval for queries
"""

from modules import memory, embedding, config_manager
from typing import List, Tuple, Dict
import numpy as np

config = config_manager.load_config()
BASE_DISTANCE = config["retrieval"]["max_distance"]

print("🔥 LOADED retrieval.py FROM:", __file__)


def _rerank_with_pronouns(query: str, results: List[Tuple[str, float, Dict]]):
    """
    Boost memories that likely resolve pronouns or named entities.
    """
    pronouns = {"he", "she", "it", "they", "him", "her", "them"}
    if not any(p in query.lower().split() for p in pronouns):
        return results

    reranked = []
    for doc, dist, meta in results:
        score = 1.0 / (1.0 + dist)
        text = (meta.get("user_prompt", "") + " " + doc).lower()

        if any(kw in text for kw in ["called", "named", "is", "has"]):
            score *= 1.25
        if any(word.istitle() for word in doc.split()):
            score *= 1.10

        reranked.append((doc, dist, meta, score))

    reranked.sort(key=lambda x: x[3], reverse=True)
    return [(doc, dist, meta) for doc, dist, meta, _ in reranked]


def retrieve_context(
    query: str,
    n_results: int = 5,
    include_meta: bool = False,
    mode: str = "contextual",
    plain: bool = False,
) -> str:
    """
    Retrieve relevant memory entries from Chroma.

    plain=True:
      - returns ONLY the most relevant factual content
      - no decorations, no metadata
      - ideal for direct factual queries ("What is the color of my cat?")
    """

    # --------------------------------------------------
    # Build safe Chroma filter
    # --------------------------------------------------
    if mode == "contextual":
        where_filter = {"intent": {"$eq": "fact"}}
    else:
        where_filter = None  # global search

    query_vector = embedding.get_embedding(query)
    if not query_vector:
        print("⚠️ Failed to generate query embedding.")
        return ""

    # --------------------------------------------------
    # Perform Chroma query
    # --------------------------------------------------
    try:
        query_kwargs = dict(
            query_embeddings=[query_vector],
            n_results=n_results,
            include=["documents", "distances", "metadatas"],
        )
        if where_filter:
            query_kwargs["where"] = where_filter

        results = memory.collection.query(**query_kwargs)
    except Exception as e:
        print(f"⚠️ Retrieval failed: {e}")
        return ""

    if not results or not results.get("documents") or not results["documents"][0]:
        print("⚠️ No matching memory found.")
        return ""

    docs = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    # --------------------------------------------------
    # Distance filtering (adaptive)
    # --------------------------------------------------
    threshold = BASE_DISTANCE
    relevant = [
        (doc, dist, meta)
        for doc, dist, meta in zip(docs, distances, metadatas)
        if dist <= threshold
    ]

    if not relevant and distances:
        avg_dist = np.mean(distances)
        new_threshold = min(avg_dist + 0.25, 1.2)
        print(
            f"⚙️ Relaxing threshold {threshold:.2f} → {new_threshold:.2f} "
            f"(avg_dist={avg_dist:.3f})"
        )
        relevant = [
            (doc, dist, meta)
            for doc, dist, meta in zip(docs, distances, metadatas)
            if dist <= new_threshold
        ]
        threshold = new_threshold

    if not relevant:
        print(f"⚠️ No relevant items under distance threshold ({threshold:.2f}).")
        return ""

    # --------------------------------------------------
    # Re-ranking
    # --------------------------------------------------
    relevant = _rerank_with_pronouns(query, relevant)

    # --------------------------------------------------
    # Plain factual mode (TOP-1, authoritative)
    # --------------------------------------------------
    if plain:
        top_doc, _, _ = relevant[0]
        print("✅ Retrieved 1 authoritative fact (plain mode)")
        return top_doc.strip()

    # --------------------------------------------------
    # Decorated / multi-context mode
    # --------------------------------------------------
    if include_meta:
        context_lines = [
            (
                f"[Meta — tag: {meta.get('tag', 'NONE')} | "
                f"date: {meta.get('timestamp', 'unknown')}]\n"
                f"User said: {meta.get('user_prompt', 'N/A')}\n"
                f"Stored output: {doc}\n"
            )
            for doc, _, meta in relevant
        ]
    else:
        context_lines = [
            (
                f"[Memory — tag: {meta.get('tag', 'NONE')} | "
                f"distance: {dist:.3f}]\n{doc}\n"
            )
            for doc, dist, meta in relevant
        ]

    print(
        f"✅ Retrieved {len(relevant)} relevant memories "
        f"(mode={mode}, distance ≤ {threshold:.2f})"
    )

    return "\n---\n".join(context_lines)
