# proxy_api/services/context_injector.py
"""
Context injector for CAM proxy:
Retrieves relevant memory, augments user prompts,
and stores new conversations to Chroma automatically.
"""

import sys, os
from datetime import datetime
from nanoid import generate

# Ensure modules path is visible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from modules import memory, retrieval, auto_tagger, usefulness_filter


def inject_context_if_relevant(user_prompt: str) -> str:
    """
    Retrieves relevant memory context for a given user prompt
    and returns an augmented prompt.
    """
    print(f"🔍 Checking for relevant context for: {user_prompt}")

    context = retrieval.retrieve_context(user_prompt)

    if context:
        print("📚 Retrieved context found — augmenting prompt...")
        augmented = f"Context:\n{context}\n\nUser: {user_prompt}"
        return augmented

    print("⚙️ No relevant memory found — sending plain prompt.")
    return user_prompt


def store_to_memory(user_prompt: str, llm_output: str, topic_continued: str = "False"):
    """
    Stores a user prompt + model response to Chroma memory if useful.
    """
    if not usefulness_filter.is_useful(user_prompt):
        print("🚫 Skipped storing trivial or meta prompt.")
        return

    episode_id = generate(size=12)
    timestamp = datetime.now().isoformat()
    tag = auto_tagger.auto_tag(user_prompt)

    metadata = {
        "timestamp": timestamp,
        "user_prompt": user_prompt,
        "tag": tag,
        "topic_continued": topic_continued,
    }

    try:
        memory.collection.add(
            ids=[episode_id],
            documents=[llm_output],
            metadatas=[metadata],
        )
        print(f"🧠 Stored episode {episode_id} (tag={tag}, continued={topic_continued})")
    except Exception as e:
        print(f"⚠️ Failed to store memory: {e}")
