# proxy_api/router.py
"""
Main chat routing logic for the universal CAM proxy.
Auto-detects LLM provider (OpenAI, Anthropic, Mistral, etc.)
based on API key prefix or model name.
"""

from fastapi import APIRouter, Request
from proxy_api.clients import provider_router

router = APIRouter()

@router.post("/v1/chat/completions")
async def chat(request: Request):
    """
    Receives a /v1/chat/completions request and routes it to the correct LLM provider.
    Compatible with OpenAI-style clients.
    """
    body = await request.json()
    messages = body.get("messages", [])
    model = body.get("model", "gpt-4o-mini")
    api_key = request.headers.get("Authorization", "").replace("Bearer ", "")

    # Get the last user message as the prompt
    prompt = messages[-1]["content"] if messages else ""

    response = provider_router.ask(prompt, api_key=api_key, model=model)

    return {
        "id": "cmpl-proxy-001",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": response}, "finish_reason": "stop"}
        ]
    }
