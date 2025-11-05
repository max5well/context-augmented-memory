"""
retrieval.py
Handles retrieval of relevant context from Chroma memory.
Uses explicit OpenAI embeddings since Chroma embedding_function is disabled.
Includes adaptive distance threshold logic.
"""

from modules import memory, embedding, config_manager
from typing import List, Tuple, Dict
import numpy as np

# Load config values
config = config_manager.load_config()
BASE_DISTANCE = config["retrieval"]["max_distance"]


def retrieve_context(query: str, n_results: int = 5, include_meta: bool = False) -> str:
    """
    Retrieves relevant memory entries for a given query.
    Dynamically adjusts distance threshold if few matches are found.
    """

    where_filter = {} if include_meta else {"intent": "fact"}

    # --- Generate query embedding manually ---
    query_vector = embedding.get_embedding(query)
    if not query_vector:
        print("⚠️ Failed to generate query embedding.")
        return ""

    # --- Perform query using explicit embeddings ---
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

    # If nothing passes the filter, relax the threshold automatically
    if not relevant and distances:
        avg_dist = np.mean(distances)
        new_threshold = min(avg_dist + 0.15, 0.9)
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

    # --- Build readable output ---
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

    print(f"✅ Retrieved {len(relevant)} relevant memories (distance ≤ {threshold:.2f})")
    return "\n---\n".join(context_lines)
