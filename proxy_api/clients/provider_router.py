"""
Routes LLM requests to the correct provider client
based on API key prefix or model name.
"""

import os
from proxy_api.clients import openai_client, anthropic_client, mistral_client, gemini_client

def detect_provider(api_key: str = None, model: str = "") -> str:
    """
    Detects which provider to use based on API key or model name.
    """
    key = (api_key or os.getenv("OPENAI_API_KEY", "")).lower()
    model = (model or "").lower()

    if key.startswith("sk-ant-") or "claude" in model:
        return "anthropic"
    elif key.startswith("mistral-") or "mistral" in model:
        return "mistral"
    elif key.startswith("gsk-") or "gemini" in model or "google" in model:
        return "gemini"
    elif key.startswith("sk-") or "gpt" in model:
        return "openai"
    else:
        return "openai"  # default fallback


def ask(prompt: str, api_key: str = None, model: str = "gpt-4o-mini") -> str:
    """
    Routes the prompt to the appropriate provider.
    """
    provider = detect_provider(api_key, model)

    if provider == "anthropic":
        return anthropic_client.ask(prompt, api_key=api_key, model=model)
    elif provider == "mistral":
        return mistral_client.ask(prompt, api_key=api_key, model=model)
    elif provider == "gemini":
        return gemini_client.ask(prompt, api_key=api_key, model=model)
    else:
        return openai_client.ask(prompt, api_key=api_key, model=model)
