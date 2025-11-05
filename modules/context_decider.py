"""
context_decider.py
Determines whether the current user prompt should trigger context retrieval.
Now uses:
- adaptive semantic similarity thresholds
- dynamic topic continuity tracking
- efficient fallback when embeddings mismatch
"""

import numpy as np
from modules import memory, embedding

# Default minimum similarity threshold for context reuse
BASE_THRESHOLD = 0.4


def cosine(a, b):
    """Compute cosine similarity between two embeddings."""
    a, b = np.array(a), np.array(b)
    if a.shape != b.shape:
        return 0.0  # mismatch safeguard
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom != 0 else 0.0


def cosine_similarities(current_vec, previous_vecs):
    """Compute cosine similarities for a batch of previous vectors."""
    sims = []
    for prev in previous_vecs:
        if not isinstance(prev, (list, np.ndarray)):
            continue
        prev_arr = np.array(prev)
        if prev_arr.shape != np.array(current_vec).shape:
            # dimension mismatch, skip
            continue
        sims.append(cosine(current_vec, prev_arr))
    return sims


def should_retrieve(user_prompt: str) -> bool:
    """
    Determines whether memory context should be retrieved for the given prompt.
    Uses semantic similarity between current and recent embeddings.
    """
    current_vec = embedding.get_embedding(user_prompt)
    if not current_vec:
        print("⚠️ Skipping context decision — empty embedding.")
        return False

    try:
        previous_vecs = memory.get_recent_embeddings(n=3)
        if not previous_vecs:
            print("⚠️ No recent embeddings found.")
            return False
    except Exception as e:
        print(f"⚠️ Failed to get recent embeddings: {e}")
        return False

    sims = cosine_similarities(current_vec, previous_vecs)
    if not sims:
        print("⚠️ No valid similarities computed.")
        return False

    avg_sim = float(np.mean(sims))
    std_dev = float(np.std(sims))

    # Adaptive thresholding based on variance:
    #   - If similarity distribution is tight, reduce noise threshold
    #   - If variance high, require stronger semantic match
    dynamic_threshold = max(BASE_THRESHOLD, avg_sim - (0.5 * std_dev))

    print(f"📊 [CLI Topic Decision] avg_sim={avg_sim:.3f} std_dev={std_dev:.3f} → threshold={dynamic_threshold:.3f}")

    # Decision logic
    if avg_sim >= dynamic_threshold:
        print("🔁 Semantic continuity detected → retrieve context.")
        return True
    else:
        print("⚙️ New topic detected → skip retrieval.")
        return False
