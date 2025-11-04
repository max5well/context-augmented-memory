# proxy_api/router.py
from fastapi import APIRouter, Request
from proxy_api.clients import provider_router
from proxy_api.services.context_injector import inject_context_if_relevant, store_to_memory
from modules import auto_tagger

router = APIRouter()

@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()

    model = body.get("model", "gpt-4o-mini")
    api_key = body.get("api_key")
    messages = body.get("messages", [])
    user_prompt = messages[-1]["content"] if messages else ""

    print(f"🧠 Incoming chat via proxy (model: {model})")

    # Step 1 — inject context from memory
    full_prompt = inject_context_if_relevant(user_prompt)

    # Step 2 — route to appropriate model provider
    llm_output = provider_router.ask(full_prompt, api_key=api_key, model=model)

    # Step 3 — store result in memory
    tag = auto_tagger.auto_tag(user_prompt)
    store_to_memory(user_prompt, llm_output, tag=tag, continued=True)

    # Step 4 — return standard OpenAI-style response
    return {
        "id": "cmpl-proxy-001",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": llm_output},
                "finish_reason": "stop",
            }
        ],
    }

# --- Memory Debug Endpoint ---
from modules import memory

@router.get("/v1/memory/debug")
async def memory_debug():
    """
    Returns a preview of the stored memories in Chroma.
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
