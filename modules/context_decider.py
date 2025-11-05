# modules/maintenance/context_decider.py

from modules import embedding, memory, config_manager, usefulness_filter
from typing import List, Tuple, Dict
import numpy as np

config = config_manager.load_config()
BASE = config["context_decider"]["continuity_base"]
STD_FACTOR = config["context_decider"]["continuity_std_factor"]

def should_continue_topic(similarity_scores: List[float]) -> bool:
    """
    Used in API context — decides whether new prompt continues the previous topic.
    """
    if not similarity_scores:
        return False

    avg = np.mean(similarity_scores)
    std_dev = np.std(similarity_scores)
    threshold = BASE + (std_dev * STD_FACTOR)

    print(f"📊 [Topic Decision] avg_sim={avg:.3f} std_dev={std_dev:.3f} → threshold={threshold:.3f}")
    return avg > threshold


def should_retrieve(user_prompt: str) -> bool:
    """
    Used in CLI mode (main.py). Decides whether to retrieve based on:
    - Embedding similarity to previous memory
    - Usefulness heuristics
    """
    current_vec = embedding.get_embedding(user_prompt)
    previous_vecs = memory.get_recent_embeddings(n=3)

    if previous_vecs is None or len(previous_vecs) == 0:
        return False

    # Compute cosine similarities
    sims = cosine_similarities(current_vec, previous_vecs)
    avg = np.mean(sims)
    std_dev = np.std(sims)
    threshold = BASE + (std_dev * STD_FACTOR)

    print(f"📊 [CLI Topic Decision] avg_sim={avg:.3f} std_dev={std_dev:.3f} → threshold={threshold:.3f}")
    return avg > threshold


def cosine_similarities(vec, others: List[List[float]]) -> List[float]:
    """Computes cosine similarities between vec and each vector in others."""
    def cosine(a, b):
        if len(a) != len(b):  # safeguard for dimension mismatch
            print(f"⚠️ Skipping embedding with mismatched dimension: {len(a)} vs {len(b)}")
            return None
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    sims = [cosine(vec, other) for other in others if other is not None]
    return [s for s in sims if s is not None]
