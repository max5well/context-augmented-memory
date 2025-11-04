# proxy_api/clients/provider_router.py
"""
Routes LLM requests to the correct provider client
based on API key prefix or model name.
"""

import os
from proxy_api.clients import openai_client, anthropic_client, mistral_client

def detect_provider(api_key: str = None, model: str = "") -> str:
    """
    Detects which provider to use based on API key or model name.
    """
    key = api_key or os.getenv("OPENAI_API_KEY", "")

    if key.startswith("sk-ant-") or "claude" in model.lower():
        return "anthropic"
    elif key.startswith("mistral-") or "mistral" in model.lower():
        return "mistral"
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
    else:
        return openai_client.ask(prompt, api_key=api_key, model=model)
