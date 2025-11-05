# proxy_api/router.py

from fastapi import APIRouter, Request
from proxy_api.clients import provider_router
from proxy_api.services.context_injector import inject_context_if_relevant, store_to_memory
from modules import auto_tagger, memory
from modules.normalizer import normalize_output

router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()

    model = body.get("model", "gpt-4o-mini")
    api_key = body.get("api_key")
    messages = body.get("messages", [])
    user_prompt = messages[-1]["content"] if messages else ""

    print(f"🧠 Incoming chat via proxy (model: {model})")

    # Step 1 — Inject memory context before prompt
    full_prompt = inject_context_if_relevant(user_prompt)

    # Step 2 — Route to appropriate provider
    llm_output = provider_router.ask(full_prompt, api_key=api_key, model=model)

    # Step 3 — Normalize output and metadata
    provider = provider_router.detect_provider(api_key=api_key, model=model)
    normalized = normalize_output(user_prompt, llm_output, model=model, provider=provider)

    # Step 4 — Extract cleaned text + full metadata
    cleaned_text = normalized.get("text", llm_output)
    metadata = normalized.get("metadata", {})

    # Step 5 — Store to memory with full metadata
    store_to_memory(user_prompt, cleaned_text, metadata=metadata)

    # Step 6 — Return OpenAI-compatible response
    return {
        "id": "cmpl-proxy-001",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": cleaned_text},
                "finish_reason": "stop",
            }
        ],
    }


# --- Memory Debug Endpoint ---
@router.get("/v1/memory/debug")
async def memory_debug():
    """
    Preview a few memory records from Chroma.
    """
    try:
        data = memory.collection.get(limit=5)
        return {
            "status": "OK",
            "count": len(data["ids"]),
            "examples": [
                {
                    "id": id_,
                    "doc": doc[:120] if doc else "",
                    "meta": meta
                }
                for id_, doc, meta in zip(data["ids"], data["documents"], data["metadatas"])
            ],
        }
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}
