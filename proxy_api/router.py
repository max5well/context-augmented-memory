# proxy_api/router.py

from fastapi import APIRouter, Request
from proxy_api.clients import provider_router
from proxy_api.services.context_injector import inject_context_if_relevant, store_to_memory
from proxy_api.services.normalize_output import normalize_response, fallback_normalize
from modules import auto_tagger, memory

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

    # Step 3 — Normalize response structure
    normalized = normalize_response(llm_output)
    if not normalized:
        print("⚠️ Output normalization failed. Running fallback model...")
        normalized = fallback_normalize(llm_output, user_prompt)
        print("⚠️ Admin alert: Output required fallback normalization. Please review.")

    # Step 4 — Extract cleaned text + metadata
    cleaned_text = normalized.get("text", llm_output)
    metadata = normalized.get("metadata", {})

    # Auto-tag fallback if none set
    if not metadata.get("tag"):
        metadata["tag"] = auto_tagger.auto_tag(user_prompt)

    # Step 5 — Store in memory
    store_to_memory(user_prompt, cleaned_text, tag=metadata["tag"], continued=metadata.get("topic_continued", True))

    # Step 6 — Return OpenAI-style response
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
