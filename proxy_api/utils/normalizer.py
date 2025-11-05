# proxy_api/utils/normalizer.py

import json
import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

FALLBACK_LLM_MODEL = os.getenv("OPENAI_API_MODEL", "gpt-3.5-turbo")
FALLBACK_LLM_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=FALLBACK_LLM_KEY)


def normalize_response_with_fallback(raw_response: str, model: str, prompt: str) -> tuple[str, dict]:
    """
    Attempts to extract the assistant message and metadata from a raw LLM response.
    Falls back to LLM-based interpretation if the structure is unknown.
    Returns: (normalized_content: str, metadata: dict)
    """
    # Case 1: known OpenAI-compatible format
    try:
        data = json.loads(raw_response) if isinstance(raw_response, str) else raw_response

        if isinstance(data, dict) and "choices" in data:
            content = data["choices"][0]["message"]["content"]
            metadata = {
                "model": data.get("model"),
                "provider": detect_provider_from_model(data.get("model", "")),
                "raw": data
            }
            return content, metadata

    except Exception:
        pass  # fallback below

    # Case 2: unknown format — use fallback LLM to parse it
    try:
        system_prompt = f"""
You are a normalization engine. Your job is to take raw LLM responses and extract:
1. The assistant's main reply message (as clean string).
2. Metadata like model name, provider if available.

Return a JSON like:
{{ "content": "...", "metadata": {{ "model": "...", "provider": "...", "notes": "..." }} }}
"""

        messages = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": f"Prompt: {prompt}\n\nRaw Response:\n{raw_response}"}
        ]

        print("⚠️ Unknown format — triggering fallback normalizer")

        completion = client.chat.completions.create(
            model=FALLBACK_LLM_MODEL,
            messages=messages,
            temperature=0.2,
        )

        result = completion.choices[0].message.content
        parsed = json.loads(result)

        return parsed.get("content", ""), parsed.get("metadata", {})

    except Exception as e:
        print(f"❌ Fallback normalization failed: {e}")
        return str(raw_response), {"fallback": True, "error": str(e)}


def detect_provider_from_model(model: str) -> str:
    """
    Maps model name patterns to known providers.
    """
    model = model.lower()

    if model.startswith("gpt-") or model.startswith("text-"):
        return "openai"
    elif model.startswith("claude-"):
        return "anthropic"
    elif model.startswith("mistral-"):
        return "mistral"
    elif "gemini" in model:
        return "gemini"
    else:
        return "unknown"
