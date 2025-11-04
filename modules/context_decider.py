"""
modules/context_decider.py
Decides whether to perform memory retrieval based on semantic similarity.
"""

from modules import memory
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# same embedder model as in memory.py
embedder = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Threshold for determining if conversation is "related"
CONTINUITY_THRESHOLD = 0.45  # smaller = stricter


def should_retrieve(current_prompt: str, lookback_n: int = 3) -> bool:
    """
    Compare the current prompt to the N most recent stored user prompts.
    If similarity is high (distance <= threshold), retrieval is warranted.
    """
    # ❌ Removed "ids" — only include what Chroma supports
    results = memory.collection.get(include=["documents", "metadatas"])

    if not results or not results.get("documents"):
        # no past data to compare
        return True

    # Get last N stored entries
    docs = results["documents"]
    if isinstance(docs[0], list):
        docs = docs[0]
    recent_docs = docs[-lookback_n:] if len(docs) > lookback_n else docs

    # Embed both
    current_emb = embedder([current_prompt])
    past_embs = embedder(recent_docs)

    # Compute cosine similarities manually
    import numpy as np
    sims = []
    for past in past_embs:
        sim = np.dot(current_emb[0], past) / (np.linalg.norm(current_emb[0]) * np.linalg.norm(past))
        sims.append(sim)

    max_sim = max(sims) if sims else 0
    print(f"🔍 Context similarity to recent memory: {max_sim:.3f}")

    return max_sim >= (1 - CONTINUITY_THRESHOLD)
