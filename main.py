"""
main.py
Standalone CLI runner for Context-Augmented Memory (CAM)
Now explicitly passes embedding vectors to Chroma after disabling auto-embedding.
"""

import os
import datetime
from openai import OpenAI
from modules import (
    embedding,
    memory,
    retrieval,
    auto_tagger,
    intent_classifier,
    context_decider,
    config_manager,
)
from uuid import uuid4

# --- Initialize client ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
config = config_manager.load_config()

print("🧠 Context-Augmented Memory System (CAM)")
print("Type 'exit' to quit or 'clear memory' to reset stored context.\n")


def main():
    while True:
        user_prompt = input("Enter your prompt: ").strip()
        if not user_prompt:
            continue

        if user_prompt.lower() in ["exit", "quit"]:
            break
        if user_prompt.lower() in ["clear memory", "reset"]:
            os.system("rm -rf CAM_project/chroma_db")
            print("🧹 Memory cleared.")
            continue

        # Step 1 — Decide whether to use memory retrieval
        try:
            should_use_context = context_decider.should_retrieve(user_prompt)
        except Exception as e:
            print(f"⚠️ Retrieval decision failed: {e}")
            should_use_context = False

        if should_use_context:
            print("🔎 Semantic continuity detected — retrieving context...\n")
            context = retrieval.retrieve_context(user_prompt)
            full_prompt = f"{context}\n\nUser: {user_prompt}"
        else:
            print("⚙️ New topic detected — skipping retrieval.")
            full_prompt = user_prompt

        # Step 2 — Send to LLM
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

        print(f"\n🤖 LLM Output:\n {llm_output}\n")

        # Step 3 — Intent and tagging
        intent = intent_classifier.classify_intent(user_prompt)
        if intent in ["meta", "query"]:
            print("🚫 Skipped storing trivial, query, or meta prompt.")
            print("------------------------------------------------------------\n")
            continue

        tag = auto_tagger.auto_tag(user_prompt)

        # Step 4 — Prepare metadata
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
            "topic_continued": should_use_context,
        }

        # Step 5 — Generate embedding and store in Chroma
        try:
            embedding_vector = embedding.get_embedding(llm_output)
            memory.store(llm_output, meta, embedding_vector)
        except Exception as e:
            print(f"❌ Failed to store memory: {e}")

        print(f"🧠 Episode {episode_id} stored (tag: {tag}, continued: {should_use_context})")
        print("------------------------------------------------------------\n")


if __name__ == "__main__":
    main()
