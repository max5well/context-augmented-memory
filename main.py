"""
main.py
Standalone CLI runner for Context-Augmented Memory (CAM)

FINAL VERSION:
- Facts are written ONLY by the user
- Queries NEVER write memory
- Facts are NEVER augmented with context
- Queries use authoritative FACTS prompt
- LLM output can never pollute memory
"""

import os
import datetime
from uuid import uuid4
from openai import OpenAI

from modules import (
    embedding,
    memory,
    retrieval,
    auto_tagger,
    intent_classifier,
    config_manager,
)

# --------------------------------------------------
# Initialization
# --------------------------------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
config = config_manager.load_config()

print("🧠 Context-Augmented Memory System (CAM)")
print("Type 'exit' to quit or 'clear memory' to reset stored context.\n")


def sanitize_metadata(meta: dict) -> dict:
    """Ensure metadata is Chroma-safe."""
    safe_meta = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float)) or v is None:
            safe_meta[k] = v
        else:
            safe_meta[k] = str(v)
    return safe_meta


def main():
    while True:
        user_prompt = input("Enter your prompt: ").strip()
        if not user_prompt:
            continue

        if user_prompt.lower() in {"exit", "quit"}:
            break

        if user_prompt.lower() in {"clear memory", "reset"}:
            os.system("rm -rf CAM_project/chroma_db")
            print("🧹 Memory cleared.")
            continue

        # --------------------------------------------------
        # Step 1 — Intent detection
        # --------------------------------------------------
        intent = intent_classifier.classify_intent(user_prompt)
        print(f"🎯 Detected intent: {intent}")

        # --------------------------------------------------
        # Step 2 — Retrieval (READ-ONLY)
        # --------------------------------------------------
        context = ""

        if intent == "query":
            print("🔍 Query detected — searching memory globally...")
            context = retrieval.retrieve_context(
                user_prompt,
                n_results=1,
                mode="global",
                plain=True,
            )

        should_use_context = bool(context)

        # --------------------------------------------------
        # Step 3 — Prompt construction (CRITICAL FIX)
        # --------------------------------------------------

        if intent == "query" and should_use_context:
            print("📚 Retrieved context found — augmenting your prompt..\n")

            full_prompt = (
                "You are answering using the following known facts.\n"
                "These facts are true and must be used to answer the question.\n\n"
                f"FACTS:\n{context}\n\n"
                f"QUESTION:\n{user_prompt}\n\n"
                "Answer using only the facts above."
            )

        else:
            # IMPORTANT:
            # - facts are sent RAW
            # - facts are NOT augmented
            # - chat/meta stay unchanged
            full_prompt = user_prompt

        # --------------------------------------------------
        # Step 4 — LLM generation
        # --------------------------------------------------
        print("💬 Sending prompt to LLM...\n")
        try:
            response = client.responses.create(
                model="gpt-4o-mini",
                input=full_prompt,
            )
            llm_output = response.output_text.strip()
        except Exception as e:
            print(f"❌ LLM request failed: {e}")
            llm_output = "(Error: LLM request failed)"

        print(f"\n🤖 LLM Output:\n{llm_output}\n")

        # --------------------------------------------------
        # Step 5 — NEVER store query answers
        # --------------------------------------------------
        if intent == "query":
            print("🚫 Query intent — skipping memory storage.")
            print("------------------------------------------------------------\n")
            continue

        # --------------------------------------------------
        # Step 6 — Store FACTS ONLY (USER INPUT)
        # --------------------------------------------------
        tag = auto_tagger.auto_tag(user_prompt)
        episode_id = str(uuid4())[:12]

        meta = {
            "episode_id": episode_id,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "session_id": f"sess_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "schema_version": "2025-11",
            "user_prompt": user_prompt,
            "tag": tag,
            "intent": intent,
        }

        meta = sanitize_metadata(meta)

        try:
            # FACTS are stored EXACTLY as the user said them
            embedding_vector = embedding.get_embedding(user_prompt)
            memory.store(user_prompt, meta, embedding_vector)
            print(f"🧠 Stored fact: {episode_id} ({len(embedding_vector)} dims) ✅")
        except Exception as e:
            print(f"❌ Failed to store memory: {e}")

        print(f"🧠 Episode {episode_id} stored (tag: {tag}, intent: {intent})")
        print("------------------------------------------------------------\n")


if __name__ == "__main__":
    main()
