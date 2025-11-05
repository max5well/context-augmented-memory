"""
embedding.py
Handles embeddings and similarity calculations.
"""

import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text: str, model: str = "text-embedding-3-small") -> list:
    """
    Returns embedding vector for a given text.
    """
    if not text.strip():
        return []

    try:
        response = client.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"⚠️ Embedding generation failed: {e}")
        return []


def cosine_similarity(a: list, b: list) -> float:
    """
    Compute cosine similarity between two embedding vectors.
    """
    a, b = np.array(a), np.array(b)
    if np.all(a == 0) or np.all(b == 0):
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
