# proxy_api/utils/normalizer.py

import uuid
from datetime import datetime
from proxy_api.utils.fallback_llm import recover_response_format  # You’ll create this next

def normalize_response(raw, provider="openai", prompt="(unknown)") -> dict:
    """
    Normalize different LLM response formats to a consistent structure.
    Adds full metadata block for storage in CAM memory.
    """
    now = datetime.utcnow().isoformat()

    try:
        # === Try OpenAI/Anthropic/Mistral ===
        if provider in ["openai", "mistral"]:
            message = raw["choices"][0]["message"]
            content = message["content"]
        elif provider == "anthropic":
            content = raw["completion"]
        elif provider == "gemini":
            content = raw["candidates"][0]["content"]["parts"][0]["text"]
        else:
            raise ValueError("Unknown provider")

        return {
            "prompt": prompt,
            "response": content.strip(),
            "metadata": {
                "episode_id": f"ep-{uuid.uuid4().hex[:10]}",
                "timestamp": now,
                "provider": provider,
                "model": raw.get("model", "unknown"),
                "tag": "UNKNOWN",
                "intent": "UNKNOWN",
                "topic_continued": True,
                "reward": 0.5,
                "session_id": f"sess-{now[:10]}",

                # optional diagnostics
                "alert": None,
                "raw_response_structure": "standard",
                "usage": raw.get("usage", {}),
            }
        }

    except Exception as e:
        # === Fallback: structure via LLM ===
        fallback = recover_response_format(raw)

        return {
            "prompt": prompt,
            "response": fallback.get("response", ""),
            "metadata": {
                "episode_id": f"ep-{uuid.uuid4().hex[:10]}",
                "timestamp": now,
                "provider": provider,
                "model": raw.get("model", "unknown"),
                "tag": "UNKNOWN",
                "intent": "UNKNOWN",
                "topic_continued": True,
                "reward": 0.0,
                "session_id": f"sess-{now[:10]}",

                "alert": f"⚠️ Unexpected format for provider: {provider} — Admin should verify",
                "raw_response_structure": "recovered",
                "error": str(e),
                "raw": str(raw)[:500]  # truncate for safety
            }
        }
