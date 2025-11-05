"""
main.py
Standalone CLI runner for Context-Augmented Memory (CAM)
Enhanced with intent-aware logic and dual-mode memory retrieval.
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
    context_decider,
    config_manager,
)

# --- Initialize ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
config = config_manager.load_config()

print("🧠 Context-Augmented Memory System (CAM)")
print("Type 'exit' to quit or 'clear memory' to reset stored context.\n")


def main():
    while True:
        user_prompt = input("Enter your prompt: ").strip()
        if not user_prompt:
            continue

        # --- Exit / Clear memory ---
        if user_prompt.lower() in ["exit", "quit"]:
            break
        if user_prompt.lower() in ["clear memory", "reset"]:
            os.system("rm -rf CAM_project/chroma_db")
            print("🧹 Memory cleared.")
            continue

        # --- Step 1: Classify intent ---
        intent = intent_classifier.classify_intent(user_prompt)
        print(f"🎯 Detected intent: {intent}")

        # --- Step 2: Decide retrieval mode ---
        context = ""

        if intent == "fact":
            print("🧩 Detected factual input — will store after LLM response.")
        elif intent == "query":
            print("🔍 Query detected — searching memory globally...")
            context = retrieval.retrieve_context(
                user_prompt, n_results=5, include_meta=True, mode="global"
            )
        elif intent == "meta":
            print("🧭 Meta command — skipping retrieval.")
        else:
            # Default: use continuity detection
            try:
                should_use_context = context_decider.should_retrieve(user_prompt)
            except Exception as e:
                print(f"⚠️ Retrieval decision failed: {e}")
                should_use_context = False

            if should_use_context:
                print("🔎 Semantic continuity detected — retrieving context...\n")
                context = retrieval.retrieve_context(
                    user_prompt, n_results=5, mode="continuity"
                )
            else:
                print("⚙️ New topic detected — trying global recall...")
                context = retrieval.retrieve_context(
                    user_prompt, n_results=5, include_meta=True, mode="global"
                )

        if context:
            print("📚 Retrieved context found — augmenting your prompt...\n")
            full_prompt = f"{context}\n\nUser: {user_prompt}"
        else:
            print("🚫 No relevant context found — proceeding without memory.")
            full_prompt = user_prompt

        # --- Step 3: Send to LLM ---
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

        # --- Step 4: Skip storing if meta or query intent ---
        if intent in ["meta", "query"]:
            print("🚫 Skipped storing trivial, query, or meta prompt.")
            print("------------------------------------------------------------\n")
            continue

        # --- Step 5: Tagging ---
        tag = auto_tagger.auto_tag(user_prompt)

        # --- Step 6: Prepare metadata ---
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
            "topic_continued": intent != "fact" and context != "",
        }

        # --- Step 7: Generate embedding & store ---
        try:
            embedding_vector = embedding.get_embedding(llm_output)
            memory.store(llm_output, meta, embedding_vector)
        except Exception as e:
            print(f"❌ Failed to store memory: {e}")

        print(f"🧠 Episode {episode_id} stored (tag: {tag}, intent: {intent})")
        print("------------------------------------------------------------\n")


if __name__ == "__main__":
    main()
