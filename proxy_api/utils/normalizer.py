# modules/normalizer.py

from datetime import datetime
import uuid
import os

from modules import auto_tagger, intent_classifier, topic_extractor

def generate_session_id():
    """
    Example: sess_2025_11_05_1530
    """
    return "sess_" + datetime.utcnow().strftime("%Y_%m_%d_%H%M")

def normalize_output(user_prompt: str, llm_response: str, model: str = "unknown", provider: str = "openai") -> dict:
    """
    Converts the raw LLM output + metadata into normalized CAM format.
    """
    timestamp = datetime.utcnow().isoformat()
    episode_id = str(uuid.uuid4())[:12]  # shorter UUID

    try:
        tag = auto_tagger.auto_tag(user_prompt)
    except Exception:
        tag = "NONE"

    try:
        intent = intent_classifier.classify_intent(user_prompt)
    except Exception:
        intent = "unknown"

    try:
        topic = topic_extractor.extract_topic(user_prompt)
    except Exception:
        topic = "unknown"

    # --- Optionally, fill these in later ---
    usage = {
        "input_tokens": None,
        "output_tokens": None,
        "latency_ms": None
    }

    metadata = {
        # ---- Identification ----
        "episode_id": episode_id,
        "timestamp": timestamp,
        "session_id": generate_session_id(),
        "provider": provider,
        "model": model,
        "schema_version": "2025-11",

        # ---- Semantic attributes ----
        "user_prompt": user_prompt,
        "response_summary": llm_response[:100],
        "tag": tag,
        "intent": intent,
        "topic": topic,

        # ---- Retrieval optimization ----
        "topic_continued": True,
        "embedding_vector": None,
        "reward": None,
        "confidence": None,
        "relevance_decay": 0.02,

        # ---- Provider + usage ----
        "usage": usage,

        # ---- Contextual linkage ----
        "linked_episodes": [],
        "source": "CAM Proxy",

        # ---- Error handling / monitoring ----
        "recovered_via_llm": False,
        "alert": None,
        "error": None,
    }

    return {
        "text": llm_response,
        "metadata": metadata
    }
